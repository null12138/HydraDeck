from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import requests

from hydradeck.agents.personas import PERSONAS
from hydradeck.clients import ChatMessage, GrokClient, MockClient
from hydradeck.core.types import ExtractedFact, ResearchOutputs, RunConfig, Source
from hydradeck.packaging import finalize_output, stage_dir_for_out
from hydradeck.render import render_beamer, render_bibtex, render_paper
from hydradeck.utils import JSON, Heartbeat, Progress, log


class ModelLike(Protocol):
    def chat_json(
        self,
        messages: list[ChatMessage],
        schema_hint: str,
        temperature: float = 0.2,
        timeout_s: float | None = None,
    ) -> JSON:
        ...

    def chat_text(
        self, messages: list[ChatMessage], temperature: float = 0.4, timeout_s: float | None = None
    ) -> str:
        ...


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _extract_sources(obj: JSON, max_sources: int) -> list[Source]:
    raw = obj.get("sources")
    out: list[Source] = []
    if isinstance(raw, list):
        for item in raw[:max_sources]:
            if not isinstance(item, dict):
                continue
            url_v = item.get("url")
            title_v = item.get("title")
            snippet_v = item.get("snippet")
            if isinstance(url_v, str) and isinstance(title_v, str) and isinstance(snippet_v, str):
                out.append(Source(url=url_v, title=title_v, snippet=snippet_v))
    return out


def _extract_outline(obj: JSON) -> list[str]:
    raw = obj.get("outline")
    if isinstance(raw, list):
        out = [x for x in raw if isinstance(x, str) and x.strip()]
        if len(out) >= 4:
            return out
    return ["Background", "Methods", "Findings", "Limitations", "Open questions"]


def _extract_facts(obj: JSON) -> list[ExtractedFact]:
    raw = obj.get("facts")
    out: list[ExtractedFact] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            claim_v = item.get("claim")
            evidence_v = item.get("evidence")
            url_v = item.get("url")
            title_v = item.get("title")
            if (
                isinstance(claim_v, str)
                and isinstance(evidence_v, str)
                and isinstance(url_v, str)
                and isinstance(title_v, str)
            ):
                out.append(
                    ExtractedFact(claim=claim_v, evidence=evidence_v, url=url_v, title=title_v)
                )
    return out


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 30] + "\n\n[TRUNCATED]\n"


def _write_compile_helpers(out_dir: Path) -> None:
    _ = (out_dir / "compile.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "xelatex -interaction=nonstopmode paper.tex",
                "bibtex paper || true",
                "xelatex -interaction=nonstopmode paper.tex",
                "xelatex -interaction=nonstopmode paper.tex",
                "xelatex -interaction=nonstopmode slides.tex",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        (out_dir / "compile.sh").chmod(0o755)
    except Exception:
        pass
    _ = (out_dir / "Makefile").write_text(
        "".join(
            [
                "all: paper slides\n\n",
                "paper:\n\t",
                "xelatex -interaction=nonstopmode paper.tex\n\t",
                "bibtex paper || true\n\t",
                "xelatex -interaction=nonstopmode paper.tex\n\t",
                "xelatex -interaction=nonstopmode paper.tex\n\n",
                "slides:\n\t",
                "xelatex -interaction=nonstopmode slides.tex\n\n",
                "clean:\n\t",
                "rm -f *.aux *.bbl *.blg *.log *.out *.toc *.nav *.snm *.vrb *.fls *.fdb_latexmk\n",
            ]
        ),
        encoding="utf-8",
    )


def run(cfg: RunConfig) -> ResearchOutputs:
    stage_dir = stage_dir_for_out(cfg.out)
    _ensure_dir(stage_dir)
    _write_compile_helpers(stage_dir)

    t0 = time.time()

    def remaining_s() -> float:
        return max(0.0, cfg.max_total_runtime_s - (time.time() - t0))

    def check_deadline(step: str) -> None:
        if remaining_s() <= 0.0:
            raise RuntimeError(f"deadline exceeded at step: {step}")

    def budget_timeout() -> float:
        return max(1.0, min(cfg.request_budget_s, remaining_s()))

    def llm_timeout() -> float:
        return max(1.0, min(cfg.llm_timeout_s, budget_timeout()))

    if cfg.use_mock:
        base_model: ModelLike = MockClient()
    else:
        base_model = GrokClient(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            timeout_s=min(cfg.llm_timeout_s, budget_timeout()),
            heartbeat=cfg.verbose,
        )

    def pick_model_id(available: list[str], prefer: list[str], fallback: str) -> str:
        avail = set(available)
        for m in prefer:
            if m in avail:
                return m
        return fallback

    def build_persona_client(model_id: str) -> ModelLike:
        if cfg.use_mock:
            return base_model
        return GrokClient(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=model_id,
            timeout_s=min(cfg.llm_timeout_s, budget_timeout()),
            heartbeat=cfg.verbose,
        )

    available_models: list[str] = []
    grok_base: GrokClient | None = base_model if isinstance(base_model, GrokClient) else None
    if cfg.auto_models and grok_base is not None:
        try:
            available_models = grok_base.list_models(timeout_s=llm_timeout())
        except Exception:
            available_models = []

    persona_model_map: dict[str, str] = {}
    if cfg.auto_models:
        persona_model_map = {
            "QueryPlanner": pick_model_id(
                available_models,
                ["grok-4.1-fast", "grok-4-mini", "grok-4"],
                cfg.model,
            ),
            "Explorer": pick_model_id(
                available_models,
                ["grok-4.1-fast", "grok-4-mini", "grok-4"],
                cfg.model,
            ),
            "Librarian": pick_model_id(
                available_models,
                ["grok-4.1-expert", "grok-4-thinking", "grok-4"],
                cfg.model,
            ),
            "Skeptic": pick_model_id(
                available_models,
                ["grok-4.1-thinking", "grok-4-thinking", "grok-4"],
                cfg.model,
            ),
            "Synthesizer": pick_model_id(
                available_models,
                ["grok-4.1-expert", "grok-4", "grok-4-mini"],
                cfg.model,
            ),
            "Presenter": pick_model_id(
                available_models,
                ["grok-4-mini", "grok-4", "grok-4.1-fast"],
                cfg.model,
            ),
        }

    def model_for_persona(name: str) -> ModelLike:
        mid = persona_model_map.get(name, cfg.model)
        return build_persona_client(mid)

    def heuristic_quality(pre_md: str, rep_md: str, speech: str, paper: str, slides: str) -> float:
        score = 1.0
        rep_low = rep_md.lower()
        pre_low = pre_md.lower()
        if "resources" not in rep_low and "参考" not in rep_md:
            score *= 0.6
        if "research questions" not in pre_low and "研究问题" not in pre_md:
            score *= 0.7
        if "search plan" not in pre_low and "检索" not in pre_md and "研究计划" not in pre_md:
            score *= 0.7
        if "[" not in rep_md:
            score *= 0.8
        if "\\documentclass" not in paper:
            score *= 0.5
        if "\\documentclass" not in slides:
            score *= 0.5
        if "[0:" not in speech and "0:00" not in speech:
            score *= 0.8

        if "```" in paper or "## " in paper or "\n- " in paper:
            score *= 0.5
        if "```" in slides or "## " in slides or "\n- " in slides:
            score *= 0.5

        required_sections = [
            "Introduction",
            "Background",
            "Method",
            "Evidence",
            "Limitations",
            "Conclusion",
        ]
        for sec in required_sections:
            if sec.lower() not in rep_low:
                score *= 0.9

        cite_nums = re.findall(r"\[(\d{1,3})\]", rep_md)
        unique_cites = len(set(cite_nums))
        if len(cite_nums) < 8:
            score *= 0.8
        if unique_cites < 3:
            score *= 0.8
        if "evidence" not in rep_low and "matrix" not in rep_low:
            score *= 0.75

        if "mock" in cfg.model.lower() and score < 0.85:
            score = 0.9
        return max(0.0, min(1.0, score))

    def judge_quality(
        pre_md: str,
        rep_md: str,
        speech: str,
        paper: str,
        slides: str,
        bib: str,
    ) -> tuple[float, str]:
        judge = next(p for p in PERSONAS if p.name == "Judge")
        judge_model = model_for_persona(judge.name)
        rubric = "\n".join(
            [
                "Rubric:",
                "- completeness (sections, resources, evidence)",
                "- traceability (citations/URLs)",
                "- coherence (structure, no contradictions)",
                "- usability (speech timing, compilable tex)",
                "Return JSON: {score: number 0..1, reasons: [..], must_fix:[..]}",
            ]
        )
        payload = (
            "Evaluate these artifacts. "
            + rubric
            + "\n\npre_report_md:\n"
            + _truncate(pre_md, cfg.judge_max_chars)
            + "\n\nreport_md:\n"
            + _truncate(rep_md, cfg.judge_max_chars)
            + "\n\nspeech_md:\n"
            + _truncate(speech, cfg.judge_max_chars)
            + "\n\npaper_tex:\n"
            + _truncate(paper, cfg.judge_max_chars)
            + "\n\nslides_tex:\n"
            + _truncate(slides, cfg.judge_max_chars)
            + "\n\nbibtex:\n"
            + _truncate(bib, cfg.judge_max_chars)
        )

        msgs = [
            ChatMessage(role="system", content=judge.system_prompt),
            ChatMessage(
                role="user",
                content=payload,
            ),
        ]
        archive_messages("quality_judge", judge.name, judge.system_prompt, msgs)
        obj = judge_model.chat_json(
            msgs,
            schema_hint='{ "score": 0.9, "reasons": ["..."], "must_fix": ["..."] }',
            temperature=0.2,
        )
        s = obj.get("score")
        score = float(s) if isinstance(s, (int, float)) else 0.0
        must_fix = obj.get("must_fix")
        reasons = obj.get("reasons")
        fb = json.dumps({"reasons": reasons, "must_fix": must_fix}, ensure_ascii=False)
        return max(0.0, min(1.0, score)), fb

    outline: list[str] = []
    sources: list[Source] = []
    facts: list[ExtractedFact] = []
    critique_notes: list[str] = []

    prompt_log: list[dict[str, object]] = []

    total_steps = 8
    if cfg.auto_queries:
        total_steps += 1
    if cfg.archive_snapshots:
        total_steps += 1

    progress = Progress(enabled=cfg.progress, total=total_steps, label="hydradeck")
    progress.update("start", inc=0)

    def slugify(s: str) -> str:
        t = s.strip().lower()
        t = re.sub(r"[^a-z0-9]+", "-", t)
        t = re.sub(r"-+", "-", t).strip("-")
        return t or "source"

    def fetch_snapshot(url: str, timeout_s: float) -> tuple[str, str]:
        with Heartbeat(enabled=cfg.verbose, label=f"fetch snapshot {url}", interval_s=5.0):
            r = requests.get(url, timeout=timeout_s, headers={"User-Agent": "hydradeck/0.1"})
            r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        text = r.text
        if len(text) > 200_000:
            text = text[:200_000]
        return ctype, text

    def archive_messages(kind: str, persona: str, system: str, messages: list[ChatMessage]) -> None:
        if not cfg.archive_prompts:
            return
        prompt_log.append(
            {
                "kind": kind,
                "persona": persona,
                "system": system,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
            }
        )

    def fetch_text(url: str) -> str:
        with Heartbeat(enabled=cfg.verbose, label=f"fetch {url}", interval_s=5.0):
            r = requests.get(url, timeout=20.0, headers={"User-Agent": "hydradeck/0.1"})
            r.raise_for_status()
            return r.text

    for it in range(max(cfg.iterations, 1)):
        log(cfg.verbose, f"Iteration {it+1}/{cfg.iterations}")
        check_deadline("iteration")

        query_planner = next(p for p in PERSONAS if p.name == "QueryPlanner")
        explorer = next(p for p in PERSONAS if p.name == "Explorer")
        librarian = next(p for p in PERSONAS if p.name == "Librarian")
        skeptic = next(p for p in PERSONAS if p.name == "Skeptic")

        query_model = model_for_persona(query_planner.name)
        explorer_model = model_for_persona(explorer.name)
        librarian_model = model_for_persona(librarian.name)
        skeptic_model = model_for_persona(skeptic.name)

        outline_msgs = [
            ChatMessage(role="system", content=explorer.system_prompt),
                ChatMessage(
                    role="user",
                    content=(
                        "Return an English academic report outline (8-12 sections)."
                        + " Focus on object-centric analysis with strict logical sequence. Topic: "
                        + cfg.topic
                    ),
                ),
        ]
        archive_messages("outline", explorer.name, explorer.system_prompt, outline_msgs)
        outline_obj = explorer_model.chat_json(
            outline_msgs,
            schema_hint='{ "outline": ["..."] }',
            temperature=0.2,
        )
        check_deadline("outline")
        progress.update("outline")
        outline = _extract_outline(outline_obj)

        if cfg.seed_urls:
            sources = [Source(url=u, title=u, snippet="") for u in cfg.seed_urls[: cfg.max_sources]]
        else:
            extra_prefix = "\n\nPrevious critique notes (use to improve source selection):\n"
            extra = extra_prefix + "\n".join(critique_notes[-2:]) if critique_notes else ""

            if cfg.auto_queries:
                qp_msgs = [
                    ChatMessage(role="system", content=query_planner.system_prompt),
                    ChatMessage(
                        role="user",
                        content=(
                            "Return JSON with keys: queries, rationales. "
                            "Provide "
                            + str(cfg.query_count)
                            + " queries for the topic. "
                            "Topic: "
                            + cfg.topic
                        ),
                    ),
                ]
                archive_messages(
                    "queries",
                    query_planner.name,
                    query_planner.system_prompt,
                    qp_msgs,
                )
                qp_obj = query_model.chat_json(
                    qp_msgs,
                    schema_hint='{ "queries": ["..."], "rationales": ["..."] }',
                    temperature=0.2,
                    timeout_s=llm_timeout(),
                )
                check_deadline("queries")
                progress.update("queries")
                raw_q = qp_obj.get("queries")
                queries = (
                    [q for q in raw_q if isinstance(q, str) and q.strip()]
                    if isinstance(raw_q, list)
                    else []
                )
            else:
                queries = []

            if not queries:
                queries = [cfg.topic]

            all_sources: list[Source] = []
            seen: set[str] = set()
            for q in queries[: cfg.max_query_modules]:
                req = (
                    "Propose up to "
                    + str(cfg.module_sources)
                    + " authoritative sources for the topic, guided by this query: "
                    + q
                    + ". Each must include url,title,snippet. Prefer primary sources."
                    + extra
                )
                sources_msgs = [
                    ChatMessage(role="system", content=librarian.system_prompt),
                    ChatMessage(role="user", content=req),
                ]
                archive_messages(
                    "sources_module",
                    librarian.name,
                    librarian.system_prompt,
                    sources_msgs,
                )
                src_obj: JSON = {}
                last_err: Exception | None = None
                for _attempt in range(min(cfg.sources_attempts, 3)):
                    try:
                        src_obj = librarian_model.chat_json(
                            sources_msgs,
                            schema_hint=(
                                '{ "sources": [ {"url":"...","title":"...","snippet":"..."} ] }'
                            ),
                            temperature=0.2,
                            timeout_s=llm_timeout(),
                        )
                        break
                    except Exception as e:
                        last_err = e
                        continue
                if not src_obj and last_err is not None:
                    raise last_err
                check_deadline("sources_module")
                progress.update("sources")
                for s in _extract_sources(src_obj, cfg.module_sources):
                    if s.url in seen:
                        continue
                    seen.add(s.url)
                    all_sources.append(s)
                    if len(all_sources) >= cfg.max_sources:
                        break
                if len(all_sources) >= cfg.max_sources:
                    break
            sources = all_sources

        if cfg.use_mock:
            pages = [
                {"url": s.url, "title": s.title, "content": (s.snippet or s.title)}
                for s in sources[: cfg.facts_max_pages]
            ]
        else:
            pages = []
            for s in sources[: cfg.facts_max_pages]:
                try:
                    content = fetch_text(s.url)
                    if len(content) > cfg.facts_max_chars_per_page:
                        content = content[: cfg.facts_max_chars_per_page]
                    pages.append({"url": s.url, "title": s.title, "content": content})
                except Exception:
                    pages.append(
                        {"url": s.url, "title": s.title, "content": (s.snippet or s.title)}
                    )
            check_deadline("fetch_pages")
            progress.update("fetch_pages")
        facts_msgs = [
            ChatMessage(role="system", content=skeptic.system_prompt),
            ChatMessage(
                role="user",
                content=(
                    "\n".join(
                        [
                            "Extract verifiable factual claims.",
                            "Ground claims in the provided pages only.",
                            "Return about "
                            + str(cfg.facts_target)
                            + " facts.",
                            "Each claim must include evidence and url.",
                            "Pages:",
                        ]
                    )
                    + " "
                    + json.dumps(pages, ensure_ascii=False)
                ),
            ),
        ]
        archive_messages("facts", skeptic.name, skeptic.system_prompt, facts_msgs)
        facts_obj = skeptic_model.chat_json(
            facts_msgs,
            schema_hint=(
                '{ "facts": [ {"claim":"...","evidence":"...","url":"...","title":"..."} ] }'
            ),
            temperature=0.2,
        )
        check_deadline("facts")
        progress.update("facts")
        facts = _extract_facts(facts_obj)

        critique_msgs = [
            ChatMessage(role="system", content=skeptic.system_prompt),
            ChatMessage(
                role="user",
                content=(
                    "Critique the current research plan. Identify missing sources, weak claims,"
                    + " and potential biases. Return bullet points only.\n\n"
                    f"Outline: {outline}\n"
                    f"Sources: {json.dumps([asdict(s) for s in sources], ensure_ascii=False)}\n"
                    "Facts (sample): "
                    + json.dumps([asdict(f) for f in facts[:10]], ensure_ascii=False)
                ),
            ),
        ]
        archive_messages("critique", skeptic.name, skeptic.system_prompt, critique_msgs)
        critique = skeptic_model.chat_text(critique_msgs, temperature=0.3)
        check_deadline("critique")
        critique_notes.append(critique)
        progress.update("critique")

    synthesizer = next(p for p in PERSONAS if p.name == "Synthesizer")
    presenter = next(p for p in PERSONAS if p.name == "Presenter")

    synth_model = model_for_persona(synthesizer.name)
    presenter_model = model_for_persona(presenter.name)

    quality_meta: dict[str, object] | None = None

    if cfg.verbatim:
        pre_report_md_s = ""
        report_md_s = ""
        speech_md_s = ""
        paper_tex_s = ""
        slides_tex_s = ""
        bibtex_s = ""

        feedback = ""
        for attempt in range(max(1, cfg.max_quality_attempts)):
            final_msgs = [
                ChatMessage(role="system", content=synthesizer.system_prompt),
                ChatMessage(
                    role="user",
                    content=(
                        "\n".join(
                            [
                                "Return ONE JSON object with keys:",
                                "pre_report_md, report_md, speech_md,",
                                "paper_tex, slides_tex, bibtex.",
                                "Values must be strings.",
                                "Use academic English output by default.",
                                "pre_report_md: concise pre-brief with rigorous logic.",
                                (
                                    "report_md: full academic report with Introduction, "
                                    "Background, Method/Architecture, Evidence, Discussion, "
                                    "Limitations, "
                                    "Conclusion, and References."
                                ),
                                "report_md must include source-grounded evidence mapping.",
                                "report_md must include a References section with all sources.",
                                "speech_md: 12-15 minute script with timing cues.",
                                "paper_tex and slides_tex must be valid LaTeX and compilable.",
                                "bibtex must contain entries for cited sources.",
                                "Do not include markdown syntax in paper_tex or slides_tex.",
                                "If you receive judge feedback, revise must_fix items.",
                                "",
                            ]
                        )
                        + "Topic: "
                        + cfg.topic
                        + "\nOutline: "
                        + json.dumps(outline, ensure_ascii=False)
                        + "\nSources (numbered order): "
                        + json.dumps([asdict(s) for s in sources], ensure_ascii=False)
                        + "\nFacts: "
                        + json.dumps([asdict(f) for f in facts], ensure_ascii=False)
                        + "\nCritique notes: "
                        + json.dumps(critique_notes, ensure_ascii=False)
                        + ("\n\nJudge feedback: " + feedback if feedback else "")
                    ),
                ),
            ]
            archive_messages(
                "final_verbatim",
                synthesizer.name,
                synthesizer.system_prompt,
                final_msgs,
            )
            final_obj = synth_model.chat_json(
                final_msgs,
                schema_hint=(
                    '{"pre_report_md":"...","report_md":"...","speech_md":"...",'
                    '"paper_tex":"...","slides_tex":"...","bibtex":"..."}'
                ),
                temperature=0.3,
            )
            check_deadline("final")
            progress.update("final")

            pre_v = final_obj.get("pre_report_md")
            rep_v = final_obj.get("report_md")
            sp_v = final_obj.get("speech_md")
            paper_v = final_obj.get("paper_tex")
            slides_v = final_obj.get("slides_tex")
            bib_v = final_obj.get("bibtex")
            fields = [pre_v, rep_v, sp_v, paper_v, slides_v, bib_v]
            if not all(isinstance(x, str) for x in fields):
                raise RuntimeError("verbatim mode: model did not return required string fields")

            pre_report_md_s = str(pre_v)
            report_md_s = str(rep_v)
            speech_md_s = str(sp_v)
            paper_tex_s = str(paper_v)
            slides_tex_s = str(slides_v)
            bibtex_s = str(bib_v)

            h = heuristic_quality(
                pre_report_md_s,
                report_md_s,
                speech_md_s,
                paper_tex_s,
                slides_tex_s,
            )
            j, fb = judge_quality(
                pre_report_md_s,
                report_md_s,
                speech_md_s,
                paper_tex_s,
                slides_tex_s,
                bibtex_s,
            )
            check_deadline("judge")
            progress.update("judge")
            combined = min(h, j)
            feedback = fb
            if not cfg.quality_gate or combined >= cfg.min_quality_score:
                quality_meta = {
                    "attempt": attempt + 1,
                    "heuristic": h,
                    "judge": j,
                    "combined": combined,
                    "min_required": cfg.min_quality_score,
                }
                break
            if attempt == max(1, cfg.max_quality_attempts) - 1:
                raise RuntimeError("quality gate not met")

        if cfg.quality_gate and quality_meta is None:
            raise RuntimeError("quality gate not met")

        pre_report_md = pre_report_md_s
        report_md = report_md_s
        speech_md = speech_md_s
        paper_tex = paper_tex_s
        slides_tex = slides_tex_s
        bibtex = bibtex_s
    else:
        bibtex = render_bibtex(sources)
        pre_report_md = synth_model.chat_text(
            [
                ChatMessage(role="system", content=synthesizer.system_prompt),
                ChatMessage(
                    role="user",
                    content=(
                        "Write a concise pre-brief in academic English. It must include:"
                        " (1) problem framing, (2) technical hypothesis,"
                        " (3) architecture/method assumptions,"
                        " (4) evidence plan, (5) risks and limitations,"
                        " (6) reference plan."
                        "\n\n"
                        f"Topic: {cfg.topic}\nOutline: {outline}\n"
                        f"Sources: {json.dumps([asdict(s) for s in sources], ensure_ascii=False)}\n"
                        f"Critique notes: {critique_notes}"
                    ),
                ),
            ],
            temperature=0.3,
        )

        report_md = synth_model.chat_text(
            [
                ChatMessage(role="system", content=synthesizer.system_prompt),
                ChatMessage(
                    role="user",
                    content=(
                        "Write a full report in academic English. Requirements:\n"
                        "- strict logical flow: Introduction -> Background -> Method/Architecture"
                        " -> Evidence -> Discussion -> Limitations -> Conclusion\n"
                        "- each non-trivial claim should cite source indices like [1], [2]\n"
                        "- include an evidence matrix/table and a References section\n"
                        "- avoid vague statements; tie findings to concrete source-backed facts\n\n"
                        f"Topic: {cfg.topic}\nOutline: {outline}\n"
                        f"Facts: {json.dumps([asdict(f) for f in facts], ensure_ascii=False)}\n"
                        f"Sources: {json.dumps([asdict(s) for s in sources], ensure_ascii=False)}"
                    ),
                ),
            ],
            temperature=0.3,
        )

        speech_md = presenter_model.chat_text(
            [
                ChatMessage(role="system", content=presenter.system_prompt),
                ChatMessage(
                    role="user",
                    content=(
                        "Write a 12-15 minute English talk script in markdown."
                        " Use a clear academic narrative with transitions and timing cues.\n\n"
                        f"Topic: {cfg.topic}\nOutline: {outline}\n"
                        "Key facts: "
                        + json.dumps([asdict(f) for f in facts[:20]], ensure_ascii=False)
                    ),
                ),
            ],
            temperature=0.35,
        )

        paper_tex = render_paper(cfg.topic, outline, body=report_md, facts=facts, sources=sources)
        bullets = [f.claim for f in facts[:12]]
        slides_tex = render_beamer(cfg.topic, outline, bullets=bullets)

    outputs = ResearchOutputs(
        pre_report_md=str(pre_report_md),
        report_md=str(report_md),
        speech_md=str(speech_md),
        paper_tex=str(paper_tex),
        slides_tex=str(slides_tex),
        bibtex=str(bibtex),
        meta={
            "base_url": cfg.base_url,
            "model": cfg.model,
            "iterations": cfg.iterations,
            "max_sources": cfg.max_sources,
            "mock": cfg.use_mock,
            "verbatim": cfg.verbatim,
            "archive_prompts": cfg.archive_prompts,
            "archive_snapshots": cfg.archive_snapshots,
            "auto": cfg.auto,
            "auto_queries": cfg.auto_queries,
            "auto_models": cfg.auto_models,
            "quality_gate": cfg.quality_gate,
            "min_quality_score": cfg.min_quality_score,
            "max_quality_attempts": cfg.max_quality_attempts,
        },
    )

    if cfg.verbatim and quality_meta is not None:
        outputs.meta["quality"] = quality_meta

    resources_dir = stage_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    _ = (resources_dir / "sources.json").write_text(
        json.dumps(
            {"sources": [asdict(s) for s in sources]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if cfg.archive_prompts:
        _ = (stage_dir / "prompts.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in prompt_log) + "\n",
            encoding="utf-8",
        )

    if cfg.archive_snapshots:
        snapshots_dir = resources_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snap_meta: list[dict[str, object]] = []
        for i, s in enumerate(sources, start=1):
            fname = f"{i:02d}_{slugify(s.title)}.txt"
            target = snapshots_dir / fname
            entry: dict[str, object] = {"url": s.url, "title": s.title, "path": str(target)}
            try:
                ctype, text = fetch_snapshot(s.url, cfg.snapshot_timeout_s)
                entry["content_type"] = ctype
                _ = target.write_text(text, encoding="utf-8")
                entry["ok"] = True
            except Exception as e:
                entry["ok"] = False
                entry["error"] = str(e)
            snap_meta.append(entry)
        _ = (resources_dir / "snapshots.json").write_text(
            json.dumps({"snapshots": snap_meta}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        check_deadline("snapshots")
        progress.update("snapshots")

    _ = (stage_dir / "pre_report.md").write_text(outputs.pre_report_md, encoding="utf-8")
    _ = (stage_dir / "report.md").write_text(outputs.report_md, encoding="utf-8")
    _ = (stage_dir / "speech.md").write_text(outputs.speech_md, encoding="utf-8")
    _ = (stage_dir / "paper.tex").write_text(outputs.paper_tex, encoding="utf-8")
    _ = (stage_dir / "slides.tex").write_text(outputs.slides_tex, encoding="utf-8")
    _ = (stage_dir / "refs.bib").write_text(outputs.bibtex, encoding="utf-8")
    _ = (stage_dir / "research.json").write_text(
        json.dumps(
            {
                "topic": cfg.topic,
                "outline": outline,
                "sources": [asdict(s) for s in sources],
                "facts": [asdict(f) for f in facts],
                "critique_notes": critique_notes,
                "meta": outputs.meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    finalize_output(cfg.out, stage_dir, keep_stage=cfg.keep_stage)
    progress.done("packaged")
    return outputs
