from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)

import tempfile
import zipfile
import json
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from hydradeck.clients import ChatMessage, GrokClient
from hydradeck.config import resolve_api_key, resolve_base_url, resolve_model
from hydradeck.core.types import RunConfig
from hydradeck.pipeline import run
from hydradeck.render import (
    build_slide_frames_from_sections,
    enforce_slide_density,
    render_beamer_frames,
    render_paper,
    render_report_structured,
)


CHROME_144_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/144.0.0.0 Safari/537.36"
)

def _normalized_base_url(base_url: str) -> str:
    parsed = urlparse(base_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Base URL must start with http:// or https://")
    if not parsed.netloc:
        raise ValueError("Base URL is missing host")
    return base_url.strip().rstrip("/")


def _preflight_check(base_url: str, api_key: str, request_budget: float) -> str | None:
    if not api_key.strip():
        return "Missing API key. Fill API Key field or set GROK_API_KEY before running."

    try:
        normalized = _normalized_base_url(base_url)
    except ValueError as exc:
        return f"Invalid Base URL: {exc}"

    probe_url = f"{normalized}/v1/models"
    timeout_s = max(2.0, min(float(request_budget), 6.0))
    req = Request(
        probe_url,
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "User-Agent": CHROME_144_UA,
        },
    )

    try:
        with urlopen(req, timeout=timeout_s):
            return None
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if exc.code == 403 and "error code: 1010" in body.lower():
            return (
                "Gateway blocked this client (Cloudflare 1010), not an API-key issue. "
                "Try another network/egress IP or ask gateway admin to allow this IP."
            )
        if exc.code in {401, 403}:
            return "API key rejected (401/403). Please update GROK_API_KEY or paste a valid key."
        return f"API endpoint returned HTTP {exc.code} during preflight."
    except URLError as exc:
        return f"Cannot reach API endpoint ({probe_url}): {exc.reason}"
    except TimeoutError:
        return (
            f"API preflight timed out after {timeout_s:.0f}s. "
            "Try mock mode first, then increase Request budget."
        )


def _api_quick_check(base_url: str, api_key: str, model: str, request_budget: float) -> str:
    selected_base_url = base_url.strip() or resolve_base_url("https://api.example.com")
    selected_api_key = api_key.strip() or resolve_api_key()

    preflight_error = _preflight_check(selected_base_url, selected_api_key, request_budget)
    if preflight_error is not None:
        return f"API check failed: {preflight_error}"

    normalized = _normalized_base_url(selected_base_url)
    req_model = model.strip() or resolve_model("grok-3-mini")
    payload = {
        "model": req_model,
        "messages": [{"role": "user", "content": "reply with exactly: API_OK"}],
        "temperature": 0,
        "max_tokens": 8,
    }
    req = Request(
        f"{normalized}/v1/chat/completions",
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {selected_api_key.strip()}",
            "User-Agent": CHROME_144_UA,
            "Content-Type": "application/json",
        },
    )
    timeout_s = max(3.0, min(float(request_budget), 12.0))
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return f"API check failed: HTTP {exc.code} {text[:180]}"
    except URLError as exc:
        return f"API check failed: network error {exc.reason}"
    except TimeoutError:
        return f"API check failed: completion timeout after {timeout_s:.0f}s"

    if "API_OK" not in body:
        return f"API check uncertain: completion returned unexpected body: {body[:180]}"
    return "API check passed: models/completions reachable and auth works."


def _compile_latex_online(tex_source: str, output_name: str) -> str:
    def _compile_via_hosted_url(command: str) -> bytes:
        upload_req = Request("https://paste.rs", data=tex_source.encode("utf-8"), method="POST")
        with urlopen(upload_req, timeout=30) as upload_resp:
            hosted_url = upload_resp.read().decode("utf-8", errors="replace").strip()
        compile_from_url = (
            "https://latexonline.cc/compile?url="
            + quote(hosted_url, safe=":/?=&")
            + "&command="
            + command
            + "&force=true"
        )
        req2 = Request(compile_from_url, headers={"User-Agent": CHROME_144_UA})
        with urlopen(req2, timeout=120) as resp2:
            return resp2.read()

    errors: list[str] = []
    blob = b""
    for command in ["xelatex", "lualatex", "pdflatex"]:
        try:
            encoded = quote(tex_source, safe="")
            compile_url = (
                "https://latexonline.cc/compile?text="
                + encoded
                + "&command="
                + command
                + "&force=true"
            )
            if len(compile_url) > 6000:
                blob = _compile_via_hosted_url(command)
            else:
                req = Request(compile_url, headers={"User-Agent": CHROME_144_UA})
                with urlopen(req, timeout=90) as resp:
                    blob = resp.read()
            if blob.startswith(b"%PDF"):
                break
            blob = _compile_via_hosted_url(command)
            if blob.startswith(b"%PDF"):
                break
            errors.append(f"{command}: non-pdf response")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            errors.append(f"{command}: HTTP {exc.code} {body[:500]}")
        except Exception as exc:
            errors.append(f"{command}: {exc}")

    if not blob.startswith(b"%PDF"):
        raise RuntimeError("online renderer failed: " + " | ".join(errors[:3]))
    out_path = Path("/tmp") / output_name
    _ = out_path.write_bytes(blob)
    return str(out_path)


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if not raw:
        raise RuntimeError("empty JSON response")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("no JSON object found in response")
    parsed2 = json.loads(raw[start : end + 1])
    if not isinstance(parsed2, dict):
        raise RuntimeError("top-level JSON is not an object")
    return parsed2


def _chat_json_resilient(
    client: GrokClient,
    messages: list[ChatMessage],
    schema_hint: str,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    try:
        obj = client.chat_json(
            messages,
            schema_hint=schema_hint,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        text = client.chat_text(messages, temperature=temperature, timeout_s=timeout_s)
        return _extract_json_object(text)
    except Exception:
        return {}


def _build_stage_model_map(
    requested_model: str,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    base = requested_model.strip() or resolve_model("grok-3-mini")
    high = base
    if "mini" in base:
        high = base.replace("-mini", "")
    if high == base and base == "grok-3-mini":
        high = "grok-3"
    model_map = {
        "scope": base,
        "structure": high,
        "planner": high,
        "section": base,
        "paper": high,
        "slides": high,
    }
    if overrides:
        for key in model_map:
            v = overrides.get(key, "").strip()
            if v:
                model_map[key] = v
    return model_map


def _looks_like_template_text(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return True
    bad_markers = [
        "this section is generated",
        "no content generated",
        "lorem ipsum",
        "to be filled",
        "placeholder",
        "add key evidence-backed findings",
        "补充关键事实与证据",
    ]
    return any(m in low for m in bad_markers)


def _assert_not_template_output(module_name: str, text: str) -> None:
    if _looks_like_template_text(text):
        raise RuntimeError(f"{module_name} produced template-like content; retry required")


def _section_quality_ok(section_title: str, latex_body: str, language: str) -> bool:
    if _looks_like_template_text(latex_body):
        return False
    body = latex_body.strip()
    if len(body) < 120:
        return False
    if language == "zh":
        zh_chars = sum(1 for ch in body if "\u4e00" <= ch <= "\u9fff")
        if zh_chars < 20:
            return False
    else:
        words = [w for w in body.replace("\n", " ").split(" ") if w]
        if len(words) < 40:
            return False
    _ = section_title
    return True


def _run_agentic_pipeline(
    topic: str,
    model: str,
    base_url: str,
    api_key: str,
    request_budget: float,
    use_mock: bool,
    progress=None,
    stage_callback=None,
    language: str = "en",
    stage_models: dict[str, str] | None = None,
) -> tuple[str, str, str, str, str, str, str, str, str]:
    if not topic.strip():
        return "Topic is required.", "", "", "", "", "", "", "", ""

    selected_base_url = base_url.strip() or resolve_base_url("https://api.example.com")
    selected_api_key = api_key.strip() or resolve_api_key()
    selected_model = model.strip() or resolve_model("grok-3-mini")
    lang = language.strip().lower()
    if lang not in {"en", "zh"}:
        lang = "en"
    model_map = _build_stage_model_map(selected_model, overrides=stage_models)
    total_steps = 9
    stage_logs: list[str] = []

    def mark(step: int, label: str, detail: str) -> None:
        pct = min(max(step / total_steps, 0.0), 1.0)
        if callable(progress):
            _ = progress(pct, desc=label)
        stage_logs.append(f"{step}/{total_steps} {label}: {detail}")

    def emit_stage(
        step: int,
        label: str,
        detail: str,
        scope_text: str = "",
        section_text: str = "",
        paper_text: str = "",
        slides_text: str = "",
        pdf_paths_text: str = "",
        paper_pdf_text: str = "",
        slides_pdf_text: str = "",
    ) -> None:
        if stage_callback is None:
            return
        payload = {
            "status": f"Running: {label}",
            "progress_log": "\n".join(stage_logs),
            "scope": scope_text,
            "sections": section_text,
            "paper": paper_text,
            "slides": slides_text,
            "pdf_paths": pdf_paths_text,
            "paper_pdf": paper_pdf_text,
            "slides_pdf": slides_pdf_text,
            "progress": int(min(100, max(0, round(step / total_steps * 100)))),
            "stage": label,
            "detail": detail,
        }
        stage_callback(payload)

    mark(1, "Preflight", "checking API connectivity")
    emit_stage(1, "Preflight", "checking API connectivity")
    if not use_mock:
        preflight_error = _preflight_check(selected_base_url, selected_api_key, request_budget)
        if preflight_error is not None:
            return (
                f"Agentic run failed: {preflight_error}",
                "\n".join(stage_logs),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            )

    scope_payload: dict[str, object]
    section_plan: list[dict[str, str]]
    section_blocks: list[dict[str, str]] = []
    paper_tex = ""
    slides_tex = ""

    if use_mock:
        mark(2, "Agent-1 ScopeScout", "using mock scope")
        scope_payload = {
            "project_links": [
                {
                    "title": "RynnBrain repo",
                    "url": "https://github.com/alibaba-damo-academy/RynnBrain",
                    "reason": "Core project artifact",
                },
                {
                    "title": "arXiv references",
                    "url": "https://arxiv.org",
                    "reason": "Peer-reviewed baseline papers",
                },
            ],
            "scope": {
                "in_scope": ["architecture", "training/inference workflow", "evaluation evidence"],
                "out_scope": ["business roadmap", "non-technical marketing claims"],
                "key_questions": [
                    "What problem is solved?",
                    "What architecture choices matter?",
                    "What evidence supports claims?",
                ],
            },
        }
        emit_stage(
            2,
            "Agent-1 ScopeScout",
            "scope resolved",
            scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
        )

        mark(3, "Agent-StructureDesigner", "designing report structure")
        structure_plan = {
            "title": topic.strip(),
            "sections": [
                {"name": "Abstract", "goal": "State problem, method, key findings, and significance."},
                {"name": "Introduction", "goal": "Context, motivation, and clear research question."},
                {"name": "Methodology", "goal": "System design, assumptions, and evaluation protocol."},
                {"name": "Results", "goal": "Evidence-backed findings with explicit source links."},
                {"name": "Discussion", "goal": "Interpretation, limitations, and trade-offs."},
                {"name": "Conclusion", "goal": "Takeaways and future work."},
            ],
            "slide_style": {
                "max_bullets": 5,
                "max_words_per_bullet": 14,
                "visual_density": "low",
                "must_include": ["agenda", "method diagram slide", "results table slide", "limitations"],
            },
        }
        emit_stage(
            3,
            "Agent-StructureDesigner",
            "report structure designed",
            scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
            section_text=json.dumps(structure_plan, ensure_ascii=False, indent=2),
        )

        mark(4, "Agent-2 TemplatePlanner", "building section summaries from templates")
        section_plan = [
            {"name": "Abstract", "summary": "Concise summary of problem, method, findings, and impact."},
            {"name": "Introduction", "summary": "Problem framing and motivation in research context."},
            {"name": "Methodology", "summary": "System architecture and methodological decisions."},
            {"name": "Results", "summary": "Empirical findings and traceable evidence."},
            {"name": "Discussion", "summary": "Interpretation of findings and practical implications."},
            {"name": "Conclusion", "summary": "Actionable takeaways and next steps."},
        ]
        if lang == "zh":
            section_plan = [
                {"name": "摘要", "summary": "概述研究问题、方法、关键发现与价值。"},
                {"name": "引言", "summary": "说明背景、动机与研究问题。"},
                {"name": "方法", "summary": "阐述系统架构、方法流程与评估设置。"},
                {"name": "结果", "summary": "给出可追溯证据支持的核心结论。"},
                {"name": "讨论", "summary": "解释结果意义、局限与适用边界。"},
                {"name": "结论", "summary": "总结与后续研究建议。"},
            ]
        emit_stage(
            4,
            "Agent-2 TemplatePlanner",
            "section plan prepared",
            scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
            section_text=json.dumps({"sections": section_plan}, ensure_ascii=False, indent=2),
        )

        mark(5, "Section Agents", "drafting per-section TeX blocks")
        for sec in section_plan:
            section_blocks.append(
                {
                    "name": sec["name"],
                    "latex": (
                        f"\\subsection*{{{sec['name']}}}\n"
                        f"{sec['summary']}\\\n"
                        "Evidence should map directly to claims and include method-specific details."
                    ),
                }
            )
        emit_stage(
            5,
            "Section Agents",
            "section drafts ready",
            scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
            section_text=json.dumps({"sections": section_plan}, ensure_ascii=False, indent=2),
            paper_text="\n\n".join(block["latex"] for block in section_blocks),
        )

        mark(6, "Integrator-Paper", "merging section TeX into paper")
        paper_tex = render_report_structured(topic.strip(), section_blocks, language=lang)

        mark(7, "Integrator-Beamer", "building slide deck from report")
        frames = build_slide_frames_from_sections(section_blocks, language=lang)
        frames = enforce_slide_density(frames, language=lang)
        slides_tex = render_beamer_frames(topic.strip(), frames, language=lang)
    else:
        timeout_s = max(12.0, min(float(request_budget), 40.0))
        client_scope = GrokClient(
            base_url=selected_base_url,
            api_key=selected_api_key,
            model=model_map["scope"],
            timeout_s=timeout_s,
            max_retries=2,
            heartbeat=False,
        )
        client_structure = GrokClient(
            base_url=selected_base_url,
            api_key=selected_api_key,
            model=model_map["structure"],
            timeout_s=timeout_s,
            max_retries=2,
            heartbeat=False,
        )
        client_planner = GrokClient(
            base_url=selected_base_url,
            api_key=selected_api_key,
            model=model_map["planner"],
            timeout_s=timeout_s,
            max_retries=2,
            heartbeat=False,
        )
        client_section = GrokClient(
            base_url=selected_base_url,
            api_key=selected_api_key,
            model=model_map["section"],
            timeout_s=timeout_s,
            max_retries=2,
            heartbeat=False,
        )
        client_paper = GrokClient(
            base_url=selected_base_url,
            api_key=selected_api_key,
            model=model_map["paper"],
            timeout_s=timeout_s,
            max_retries=2,
            heartbeat=False,
        )
        client_slides = GrokClient(
            base_url=selected_base_url,
            api_key=selected_api_key,
            model=model_map["slides"],
            timeout_s=timeout_s,
            max_retries=2,
            heartbeat=False,
        )

        quick_scope = {
            "project_links": [
                {
                    "title": f"{topic.strip()} official repository",
                    "url": "https://github.com",
                    "reason": "Seed placeholder before remote scope enrichment.",
                }
            ],
            "scope": {
                "in_scope": ["architecture", "method", "evidence"],
                "out_scope": ["marketing narrative", "non-technical roadmap"],
                "key_questions": [
                    "What core problem is solved?",
                    "What design decisions matter most?",
                    "What evidence is verifiable?",
                ],
            },
        }
        emit_stage(
            2,
            "Agent-1 ScopeScout",
            "quick skeleton ready; enriching with remote call",
            scope_text=json.dumps(quick_scope, ensure_ascii=False, indent=2),
        )

        mark(2, "Agent-1 ScopeScout", "asking Grok for project links + scope")
        try:
            scope_payload = _chat_json_resilient(
                client_scope,
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "You are ScopeScout. Find key project links and define an initial technical research scope."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=(
                            "Topic: "
                            + topic.strip()
                            + "\nReturn JSON with keys: project_links (list of {title,url,reason}),"
                            + " scope ({in_scope:[...], out_scope:[...], key_questions:[...]})"
                        ),
                    ),
                ],
                schema_hint=(
                    '{"project_links":[{"title":"...","url":"https://...","reason":"..."}],'
                    '"scope":{"in_scope":["..."],"out_scope":["..."],"key_questions":["..."]}}'
                ),
                temperature=0.1,
                timeout_s=min(timeout_s, 18.0),
            )
        except Exception:
            scope_payload = quick_scope
        emit_stage(
            2,
            "Agent-1 ScopeScout",
            "scope resolved",
            scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
        )

        mark(3, "Agent-StructureDesigner", "designing report architecture and slide style")
        structure_obj = _chat_json_resilient(
            client_structure,
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You are StructureDesigner. Build a publication-grade report architecture and a presentation"
                        " style guide before drafting any sections."
                        + (" Respond in Chinese." if lang == "zh" else " Respond in English.")
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "Topic: "
                        + topic.strip()
                        + "\nScope JSON: "
                        + json.dumps(scope_payload, ensure_ascii=False)
                        + "\nReturn JSON {report_blueprint:{section_order:[...],section_goals:[...]},"
                        + " slide_style:{theme,max_bullets,max_words_per_bullet,visual_rules:[...]}}"
                        + " Ensure this is a RESEARCH REPORT structure (not academic paper IMRaD rigidity)."
                    ),
                ),
            ],
            schema_hint='{"report_blueprint":{"section_order":["..."],"section_goals":["..."]},"slide_style":{"theme":"..."}}',
            temperature=0.15,
            timeout_s=timeout_s,
        )
        if not isinstance(structure_obj, dict) or not structure_obj:
            structure_obj = {
                "report_blueprint": {
                    "section_order": [
                        "Abstract",
                        "Introduction",
                        "Methodology",
                        "Results",
                        "Discussion",
                        "Conclusion",
                    ],
                    "section_goals": [
                        "Summarize research contribution",
                        "Define context and question",
                        "Describe method rigorously",
                        "Present evidence with citations",
                        "Discuss limits and implications",
                        "Conclude and future work",
                    ],
                },
                "slide_style": {
                    "theme": "metropolis-like clean",
                    "max_bullets": 5,
                    "max_words_per_bullet": 14,
                    "visual_rules": [
                        "one idea per slide",
                        "results in table/figure frame",
                        "consistent color accents",
                    ],
                },
            }
        emit_stage(
            3,
            "Agent-StructureDesigner",
            "structure blueprint ready",
            scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
            section_text=json.dumps(structure_obj, ensure_ascii=False, indent=2),
        )

        mark(4, "Agent-2 TemplatePlanner", "mapping scope to paper/beamer section summaries")
        section_obj = _chat_json_resilient(
            client_planner,
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You are TemplatePlanner. Based on scope and LaTeX paper/beamer structure, define section"
                        " summaries that downstream section agents will write."
                        + (" Respond in Chinese." if lang == "zh" else " Respond in English.")
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "Topic: "
                        + topic.strip()
                        + "\nScope JSON: "
                        + json.dumps(scope_payload, ensure_ascii=False)
                        + "\nStructure JSON: "
                        + json.dumps(structure_obj, ensure_ascii=False)
                        + "\nReturn JSON: {sections:[{name,summary}]} with 6-8 sections for a RESEARCH REPORT."
                        + " Ensure section names are concise and audience-friendly."
                    ),
                ),
            ],
            schema_hint='{"sections":[{"name":"Introduction","summary":"..."}]}',
            temperature=0.1,
            timeout_s=timeout_s,
        )
        raw_sections = section_obj.get("sections")
        section_plan = [
            {"name": str(x.get("name", "Section")), "summary": str(x.get("summary", ""))}
            for x in raw_sections
            if isinstance(x, dict)
        ] if isinstance(raw_sections, list) else []
        section_plan = section_plan[:6]
        if not section_plan:
            section_plan = [
                {"name": "Abstract", "summary": "Concise summary of contribution and findings."},
                {"name": "Introduction", "summary": "Problem framing and objectives."},
                {"name": "Methodology", "summary": "Core architecture and methodology."},
                {"name": "Results", "summary": "Findings grounded in verifiable sources."},
            ]
        emit_stage(
            4,
            "Agent-2 TemplatePlanner",
            "section plan prepared",
            scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
            section_text=json.dumps({"sections": section_plan}, ensure_ascii=False, indent=2),
        )

        mark(5, "Section Agents", "researching each section and drafting TeX fragments")
        for idx, sec in enumerate(section_plan, start=1):
            section_title = sec["name"]
            latex_body = ""
            for attempt in range(1, 4):
                sec_obj = _chat_json_resilient(
                    client_section,
                    [
                        ChatMessage(
                            role="system",
                            content=(
                                "You are a SectionResearchAgent. Write a rigorous LaTeX fragment for your assigned"
                                " section only."
                                + (" Output Chinese text." if lang == "zh" else " Output English text.")
                            ),
                        ),
                        ChatMessage(
                            role="user",
                            content=(
                                f"Topic: {topic.strip()}\nSection: {sec['name']}\nSummary: {sec['summary']}\n"
                                f"Structure JSON: {json.dumps(structure_obj, ensure_ascii=False)}\n"
                                "Return JSON {section_title, latex_body}. latex_body must be plain LaTeX paragraphs"
                                " without documentclass/begin{document}, with evidence-driven style and citation markers."
                                " Keep each paragraph focused and concise for report readability."
                                " Minimum: 2 substantive paragraphs. No placeholder text."
                            ),
                        ),
                    ],
                    schema_hint='{"section_title":"...","latex_body":"\\subsection*{...} ..."}',
                    temperature=0.1,
                    timeout_s=timeout_s,
                )
                cand_title = sec_obj.get("section_title")
                cand_body = sec_obj.get("latex_body")
                if isinstance(cand_title, str) and cand_title.strip():
                    section_title = cand_title.strip()
                if isinstance(cand_body, str):
                    latex_body = cand_body.strip()
                if _section_quality_ok(section_title, latex_body, lang):
                    break
                emit_stage(
                    5,
                    "Section Agents",
                    f"quality gate retry {attempt}/3 for section {idx}",
                    scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
                    section_text=json.dumps({"sections": section_plan}, ensure_ascii=False, indent=2),
                    paper_text="\n\n".join(block["latex"] for block in section_blocks),
                )
            if not _section_quality_ok(section_title, latex_body, lang):
                raise RuntimeError(
                    f"Section agent failed quality gate after retries: {section_title}"
                )
            section_blocks.append({"name": section_title, "latex": latex_body})
            mark(4, "Section Agents", f"completed {idx}/{len(section_plan)} sections")
            emit_stage(
                5,
                "Section Agents",
                f"completed {idx}/{len(section_plan)} sections",
                scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
                section_text=json.dumps({"sections": section_plan}, ensure_ascii=False, indent=2),
                paper_text="\n\n".join(block["latex"] for block in section_blocks),
            )

        mark(6, "Integrator-Paper", "assembling full paper.tex")
        paper_obj = _chat_json_resilient(
            client_paper,
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You are ReportIntegrator. Produce a professional LaTeX RESEARCH REPORT"
                        " with executive readability, clear argument flow, and section coherence."
                        + (" Output Chinese text." if lang == "zh" else " Output English text.")
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "Topic: "
                        + topic.strip()
                        + "\nScope: "
                        + json.dumps(scope_payload, ensure_ascii=False)
                        + "\nStructure: "
                        + json.dumps(structure_obj, ensure_ascii=False)
                        + "\nSection snippets: "
                        + json.dumps(section_blocks, ensure_ascii=False)
                        + "\nReturn JSON {paper_tex} with a full compilable document using report sections:"
                        + " Executive Summary/Abstract, Background, Approach, Results, Discussion, Risks, Conclusion, References."
                        + " Each section should include concrete evidence statements and implementation-level details,"
                        + " not high-level filler. Minimum 2-4 substantive paragraphs per major section."
                    ),
                ),
            ],
            schema_hint='{"paper_tex":"\\documentclass{article} ... \\end{document}"}',
            temperature=0.1,
            timeout_s=timeout_s,
        )
        _paper_candidate = paper_obj.get("paper_tex")
        paper_tex = render_report_structured(topic.strip(), section_blocks, language=lang)
        _assert_not_template_output("paper", paper_tex)
        emit_stage(
            6,
            "Integrator-Paper",
            "paper.tex assembled",
            scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
            section_text=json.dumps({"sections": section_plan}, ensure_ascii=False, indent=2),
            paper_text=paper_tex,
        )

        mark(7, "Integrator-Beamer", "assembling full slides.tex")
        slides_obj = _chat_json_resilient(
            client_slides,
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You are BeamerIntegrator. Produce a visually polished, conference-style Beamer deck"
                        " with concise bullets, visual hierarchy, and readable spacing."
                        + (" Output Chinese text." if lang == "zh" else " Output English text.")
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "Topic: "
                        + topic.strip()
                        + "\nScope: "
                        + json.dumps(scope_payload, ensure_ascii=False)
                        + "\nSection plan: "
                        + json.dumps(section_plan, ensure_ascii=False)
                        + "\nSlide style: "
                        + json.dumps(structure_obj.get("slide_style", {}), ensure_ascii=False)
                        + "\nReturn JSON {slides_tex} with a full compilable beamer document."
                        + " Use modern readable typography, max 5 bullets/frame, max 14 words/bullet,"
                        + " and ensure each frame content fully fits without overflow."
                        + " Include complete coverage: agenda, background, method, results, discussion, conclusion."
                        + " Return STRICTLY compilable LaTeX without custom undefined macros."
                    ),
                ),
            ],
            schema_hint='{"slides_tex":"\\documentclass{beamer} ... \\end{document}"}',
            temperature=0.1,
            timeout_s=timeout_s,
        )
        _slides_candidate = slides_obj.get("slides_tex")
        frames = build_slide_frames_from_sections(section_blocks, language=lang)
        frames = enforce_slide_density(frames, language=lang)
        slides_tex = render_beamer_frames(topic.strip(), frames, language=lang)
        _assert_not_template_output("slides", slides_tex)
        emit_stage(
            7,
            "Integrator-Beamer",
            "slides.tex assembled",
            scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
            section_text=json.dumps({"sections": section_plan}, ensure_ascii=False, indent=2),
            paper_text=paper_tex,
            slides_text=slides_tex,
        )

    mark(8, "Online Render", "compiling paper/slides to PDF via latexonline.cc")
    emit_stage(
        8,
        "Online Render",
        "rendering started",
        scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
        section_text=json.dumps({"sections": section_plan}, ensure_ascii=False, indent=2),
        paper_text=paper_tex,
        slides_text=slides_tex,
    )
    try:
        paper_pdf = _compile_latex_online(paper_tex, "hydradeck_agentic_paper.pdf")
        slides_pdf = _compile_latex_online(slides_tex, "hydradeck_agentic_slides.pdf")
        emit_stage(
            8,
            "Online Render",
            "pdf rendered",
            scope_text=json.dumps(scope_payload, ensure_ascii=False, indent=2),
            section_text=json.dumps({"sections": section_plan}, ensure_ascii=False, indent=2),
            paper_text=paper_tex,
            slides_text=slides_tex,
            pdf_paths_text=paper_pdf + "\n" + slides_pdf,
            paper_pdf_text=paper_pdf,
            slides_pdf_text=slides_pdf,
        )
    except Exception as exc:
        return (
            f"Agentic run partial success: TeX generated but online PDF render failed: {exc}",
            "\n".join(stage_logs),
            json.dumps(scope_payload, ensure_ascii=False, indent=2),
            json.dumps({"sections": section_plan}, ensure_ascii=False, indent=2),
            paper_tex,
            slides_tex,
            "",
            "",
            "",
        )

    mark(9, "Done", "paper/slides PDFs rendered and ready")
    return (
        "Agentic pipeline done: scoped, drafted, integrated, rendered to PDF.",
        "\n".join(stage_logs),
        json.dumps(scope_payload, ensure_ascii=False, indent=2),
        json.dumps({"sections": section_plan}, ensure_ascii=False, indent=2),
        paper_tex,
        slides_tex,
        paper_pdf + "\n" + slides_pdf,
        paper_pdf,
        slides_pdf,
    )


def _run_agentic_pipeline_stream(
    topic: str,
    model: str,
    base_url: str,
    api_key: str,
    request_budget: float,
    use_mock: bool,
):
    status = "Agentic pipeline running..."
    progress_log = "1/3 Starting workflow"
    empty_json = ""
    empty_tex = ""
    empty_paths = ""
    yield (
        status,
        progress_log,
        empty_json,
        empty_json,
        empty_tex,
        empty_tex,
        empty_paths,
        "",
        "",
        5,
    )

    progress_log = "1/3 API scope and section planning"
    yield (
        status,
        progress_log,
        empty_json,
        empty_json,
        empty_tex,
        empty_tex,
        empty_paths,
        "",
        "",
        30,
    )

    events: Queue[dict[str, object]] = Queue()

    def on_stage(payload: dict[str, object]) -> None:
        events.put(payload)

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(
            _run_agentic_pipeline,
            topic,
            model,
            base_url,
            api_key,
            request_budget,
            use_mock,
            None,
            on_stage,
        )
        wait_tick = 0
        while not fut.done() or not events.empty():
            try:
                ev = events.get(timeout=1.0)
                yield (
                    str(ev.get("status", "Agentic pipeline running...")),
                    str(ev.get("progress_log", "")),
                    str(ev.get("scope", "")),
                    str(ev.get("sections", "")),
                    str(ev.get("paper", "")),
                    str(ev.get("slides", "")),
                    str(ev.get("pdf_paths", "")),
                    str(ev.get("paper_pdf", "")),
                    str(ev.get("slides_pdf", "")),
                    int(str(ev.get("progress", "0"))),
                )
                continue
            except Empty:
                pass

            wait_tick += 1
            elapsed_s = wait_tick
            heartbeat_pct = min(95, 30 + wait_tick)
            yield (
                "Agentic pipeline running...",
                f"2/3 Running agent workflow ({elapsed_s}s elapsed)",
                empty_json,
                empty_json,
                empty_tex,
                empty_tex,
                empty_paths,
                "",
                "",
                heartbeat_pct,
            )
            time.sleep(1)

        (
            status2,
            progress2,
            scope2,
            sections2,
            paper2,
            slides2,
            paths2,
            paper_pdf2,
            slides_pdf2,
        ) = fut.result()

    done_log = "3/3 Completed"
    if progress2.strip():
        done_log = progress2 + "\n" + done_log

    yield (
        status2,
        done_log,
        scope2,
        sections2,
        paper2,
        slides2,
        paths2,
        paper_pdf2,
        slides_pdf2,
        100,
    )


def _run_pipeline(
    topic: str,
    model: str,
    base_url: str,
    api_key: str,
    max_sources: int,
    iterations: int,
    llm_timeout: float,
    request_budget: float,
    seed_urls_text: str,
    use_mock: bool,
) -> tuple[str, str, str, str]:
    if not topic.strip():
        return "Topic is required.", "", "", ""

    selected_base_url = base_url.strip() or resolve_base_url("https://api.example.com")
    selected_api_key = api_key.strip() or resolve_api_key()

    if not use_mock:
        preflight_error = _preflight_check(selected_base_url, selected_api_key, request_budget)
        if preflight_error is not None:
            return f"Preflight failed: {preflight_error}", "", "", ""

    with tempfile.TemporaryDirectory() as td:
        out_zip = Path(td) / "hydradeck_out.zip"
        seeds = [x.strip() for x in seed_urls_text.splitlines() if x.strip()]
        cfg = RunConfig(
            topic=topic.strip(),
            out=out_zip,
            base_url=selected_base_url,
            api_key=selected_api_key,
            model=model.strip() or resolve_model("grok-4"),
            iterations=max(1, int(iterations)),
            max_sources=max(1, int(max_sources)),
            llm_timeout_s=float(llm_timeout),
            request_budget_s=float(request_budget),
            use_mock=bool(use_mock),
            seed_urls=seeds or None,
            progress=False,
            quality_gate=False,
            archive_snapshots=False,
        )

        retry_cfg = RunConfig(
            topic=cfg.topic,
            out=cfg.out,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            iterations=cfg.iterations,
            max_sources=cfg.max_sources,
            module_sources=cfg.module_sources,
            min_total_words=cfg.min_total_words,
            use_mock=cfg.use_mock,
            verbose=cfg.verbose,
            llm_timeout_s=max(cfg.llm_timeout_s, 90.0),
            facts_max_pages=cfg.facts_max_pages,
            facts_max_chars_per_page=cfg.facts_max_chars_per_page,
            facts_target=cfg.facts_target,
            judge_max_chars=cfg.judge_max_chars,
            pre_tex_quality_gate=cfg.pre_tex_quality_gate,
            pre_tex_min_score=cfg.pre_tex_min_score,
            pre_tex_attempts=cfg.pre_tex_attempts,
            keep_stage=cfg.keep_stage,
            verbatim=cfg.verbatim,
            archive_prompts=cfg.archive_prompts,
            archive_snapshots=cfg.archive_snapshots,
            snapshot_timeout_s=cfg.snapshot_timeout_s,
            snapshot_total_timeout_s=cfg.snapshot_total_timeout_s,
            auto=cfg.auto,
            auto_queries=cfg.auto_queries,
            auto_models=cfg.auto_models,
            quality_gate=cfg.quality_gate,
            min_quality_score=cfg.min_quality_score,
            max_quality_attempts=cfg.max_quality_attempts,
            query_count=cfg.query_count,
            max_query_modules=cfg.max_query_modules,
            sources_attempts=cfg.sources_attempts,
            max_total_runtime_s=max(cfg.max_total_runtime_s, 420.0),
            progress=cfg.progress,
            request_budget_s=max(cfg.request_budget_s, 35.0),
            pdf_compiler=cfg.pdf_compiler,
            template=cfg.template,
            seed_urls=cfg.seed_urls,
        )
        try:
            _ = run(cfg)
        except Exception as exc:
            err_text = str(exc)
            retryable = ("Read timed out" in err_text) or ("timed out" in err_text.lower())
            if (not use_mock) and retryable:
                try:
                    _ = run(retry_cfg)
                except Exception as retry_exc:
                    return (
                        "Run failed after retry: "
                        f"{retry_exc}. Try request_budget >= 35 and llm_timeout >= 90.",
                        "",
                        "",
                        "",
                    )
            else:
                return (
                    "Run failed: "
                    f"{exc}. If queue waits too long, try Use mock (offline) or increase Request budget.",
                    "",
                    "",
                    "",
                )

        with zipfile.ZipFile(out_zip, "r") as z:
            report_md = z.read("report.md").decode("utf-8", errors="replace")
            paper_tex = z.read("paper.tex").decode("utf-8", errors="replace")
            slides_tex = z.read("slides.tex").decode("utf-8", errors="replace")

        copy_zip = Path("/tmp") / "hydradeck_space_output.zip"
        copy_zip.write_bytes(out_zip.read_bytes())
        status = f"Done. Output zip: {copy_zip}"
        return status, report_md, paper_tex, slides_tex
