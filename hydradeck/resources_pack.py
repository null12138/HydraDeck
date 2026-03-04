from __future__ import annotations

import json
import re
import time
import urllib.parse
from dataclasses import asdict
from pathlib import Path

import requests

from hydradeck.agents.personas import PERSONAS
from hydradeck.clients import ChatMessage, GrokClient, GrokClientError
from hydradeck.core.types import RunConfig, Source
from hydradeck.packaging import finalize_output, stage_dir_for_out
from hydradeck.utils import Heartbeat, Progress


def _slugify(s: str) -> str:
    t = s.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = re.sub(r"-+", "-", t).strip("-")
    return t or "source"


def _extract_sources(obj: dict[str, object], max_sources: int) -> list[Source]:
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


def build_resources_pack(cfg: RunConfig) -> Path:
    stage_dir = stage_dir_for_out(cfg.out)
    stage_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    def remaining_s() -> float:
        return max(0.0, cfg.max_total_runtime_s - (time.time() - t0))

    def budget_timeout() -> float:
        return max(1.0, min(cfg.request_budget_s, remaining_s()))

    def llm_timeout() -> float:
        return max(1.0, min(cfg.llm_timeout_s, budget_timeout()))

    progress = Progress(enabled=cfg.progress, total=6, label="resources")
    progress.update("start", inc=0)

    if cfg.use_mock:
        from hydradeck.clients.grok_client import MockClient

        client = MockClient()
    else:
        client = GrokClient(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            timeout_s=llm_timeout(),
            heartbeat=cfg.verbose,
        )

    query_planner = next(p for p in PERSONAS if p.name == "QueryPlanner")
    librarian = next(p for p in PERSONAS if p.name == "Librarian")

    qp_obj = client.chat_json(
        [
            ChatMessage(role="system", content=query_planner.system_prompt),
            ChatMessage(
                role="user",
                content=(
                    "Return JSON: {queries:[...]} with 6 high-recall queries for primary sources. "
                    "Topic: "
                    + cfg.topic
                ),
            ),
        ],
        schema_hint='{ "queries": ["..."] }',
        temperature=0.2,
        timeout_s=llm_timeout() if not cfg.use_mock else None,
    )
    progress.update("queries")
    raw_q = qp_obj.get("queries")
    if isinstance(raw_q, list):
        queries = [q for q in raw_q if isinstance(q, str) and q.strip()]
    else:
        queries = []
    if not queries:
        queries = [cfg.topic]

    seen: set[str] = set()
    sources: list[Source] = []
    for q in queries[: min(3, len(queries))]:
        req = (
            "Return JSON with key sources: list of {url,title,snippet}. "
            "Give authoritative sources (prefer official docs, papers, repos). "
            "Query: "
            + q
        )
        try:
            src_obj = client.chat_json(
                [
                    ChatMessage(role="system", content=librarian.system_prompt),
                    ChatMessage(role="user", content=req),
                ],
                schema_hint='{ "sources": [ {"url":"...","title":"...","snippet":"..."} ] }',
                temperature=0.2,
                timeout_s=llm_timeout() if not cfg.use_mock else None,
            )
        except GrokClientError:
            continue
        for s in _extract_sources(src_obj, cfg.module_sources):
            if s.url in seen:
                continue
            seen.add(s.url)
            sources.append(s)
            if len(sources) >= cfg.max_sources:
                break
        if len(sources) >= cfg.max_sources:
            break
        progress.update("sources")

    if not sources:
        sources = [
            Source(
                url="https://github.com/alibaba-damo-academy/RynnBrain",
                title="RynnBrain",
                snippet="",
            )
        ]
    progress.update("sources")

    resources_dir = stage_dir / "resources"
    snaps_dir = resources_dir / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    snap_meta: list[dict[str, object]] = []
    snap_start = time.time()
    for i, s in enumerate(sources, start=1):
        if (time.time() - snap_start) > cfg.snapshot_total_timeout_s:
            break
        target_base = snaps_dir / f"{i:02d}_{_slugify(s.title)}"
        entry: dict[str, object] = {"url": s.url, "title": s.title}
        if cfg.use_mock:
            entry["ok"] = True
            target = target_base.with_suffix(".txt")
            entry["path"] = str(target)
            target.write_text("mock snapshot", encoding="utf-8")
            snap_meta.append(entry)
            continue
        try:
            with Heartbeat(enabled=cfg.verbose, label=f"fetch {s.url}", interval_s=5.0):
                r = requests.get(
                    s.url,
                    timeout=min(cfg.snapshot_timeout_s, budget_timeout()),
                    headers={"User-Agent": "hydradeck/0.1"},
                )
                r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            entry["content_type"] = ctype

            is_pdf = "application/pdf" in ctype.lower() or s.url.lower().endswith(".pdf")
            if is_pdf:
                data = r.content
                if len(data) > 5_000_000:
                    data = data[:5_000_000]
                target = target_base.with_suffix(".pdf")
                entry["path"] = str(target)
                target.write_bytes(data)
                entry["binary"] = True
            else:
                txt = r.text
                if len(txt) > 200_000:
                    txt = txt[:200_000]
                target = target_base.with_suffix(".txt")
                entry["path"] = str(target)
                target.write_text(txt, encoding="utf-8")
            entry["ok"] = True
        except Exception as e:
            entry["ok"] = False
            entry["error"] = str(e)
        snap_meta.append(entry)
    progress.update("snapshots")

    (resources_dir / "sources.json").write_text(
        json.dumps({"sources": [asdict(s) for s in sources]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (resources_dir / "snapshots.json").write_text(
        json.dumps({"snapshots": snap_meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (stage_dir / "research.json").write_text(
        json.dumps(
            {
                "topic": cfg.topic,
                "mode": "resources",
                "sources": [asdict(s) for s in sources],
                "snapshots": snap_meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    progress.update("package")

    try:
        paper_tex, slides_tex = _generate_pre_tex(cfg, client, sources)
    except Exception as e:
        (stage_dir / "pre_tex_error.txt").write_text(str(e) + "\n", encoding="utf-8")
        paper_tex = _render_paper_tex(cfg.topic, sources)
        slides_tex = _render_slides_tex(cfg.topic, sources)

    (stage_dir / "pre_paper.tex").write_text(paper_tex, encoding="utf-8")
    (stage_dir / "pre_slides.tex").write_text(slides_tex, encoding="utf-8")

    pdf_dir = stage_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    errors: list[str] = []

    if cfg.use_mock:
        (pdf_dir / "pre_paper.pdf").write_bytes(_dummy_pdf_bytes("paper"))
        (pdf_dir / "pre_slides.pdf").write_bytes(_dummy_pdf_bytes("slides"))
    else:
        try:
            paper_pdf, paper_meta = _compile_pdf(
                paper_tex,
                engine="xelatex",
                backend=cfg.pdf_compiler,
            )
            (pdf_dir / "pre_paper.pdf").write_bytes(paper_pdf)
            urls.extend(paper_meta.get("urls", []))
            errors.extend(paper_meta.get("errors", []))
        except Exception as e:
            errors.append("paper: " + str(e))

        try:
            slides_pdf, slides_meta = _compile_pdf(
                slides_tex,
                engine="xelatex",
                backend=cfg.pdf_compiler,
            )
            (pdf_dir / "pre_slides.pdf").write_bytes(slides_pdf)
            urls.extend(slides_meta.get("urls", []))
            errors.extend(slides_meta.get("errors", []))
        except Exception as e:
            errors.append("slides: " + str(e))

    if not (pdf_dir / "pre_paper.pdf").exists():
        errors.append("paper pdf missing")
    if not (pdf_dir / "pre_slides.pdf").exists():
        errors.append("slides pdf missing")

    if urls:
        (stage_dir / "latexonline_url.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
    if errors:
        (stage_dir / "latexonline_error.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")

    finalize_output(cfg.out, stage_dir, keep_stage=cfg.keep_stage)
    progress.done("packaged")
    return cfg.out


def _render_paper_tex(topic: str, sources: list[Source]) -> str:
    def esc(s: str) -> str:
        return (
            s.replace("\\", r"\textbackslash{}")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("#", r"\#")
            .replace("$", r"\$")
        )

    items = []
    for _i, s in enumerate(sources, start=1):
        items.append(
            "\\item "
            + esc(s.title)
            + "\\\\\n"
            + "\\small\\url{" + esc(s.url) + "}\\normalsize\\\\\n"
            + "\\textit{" + esc(s.snippet[:240]) + "}"
        )
    body = "\n".join(items) if items else "\\item （暂无来源）"
    return (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[UTF8]{ctex}\n"
        "\\usepackage{hyperref}\n"
        "\\usepackage{url}\n"
        "\\usepackage{booktabs}\n"
        "\\title{" + esc(topic) + "——资源预研报告（论文版）}\n"
        "\\author{hydradeck}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\section*{来源清单}\n"
        "\\begin{enumerate}\n"
        + body
        + "\n\\end{enumerate}\n"
        "\\end{document}\n"
    )


def _render_slides_tex(topic: str, sources: list[Source]) -> str:
    def esc(s: str) -> str:
        return (
            s.replace("\\", r"\textbackslash{}")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("#", r"\#")
            .replace("$", r"\$")
        )

    bullets: list[str] = []
    for s in sources[:8]:
        bullets.append(esc(s.title))

    items = "\n".join(["\\item " + b for b in bullets]) or "\\item （暂无来源）"
    return (
        "\\documentclass{beamer}\n"
        "\\usepackage[UTF8]{ctex}\n"
        "\\usetheme{Madrid}\n"
        "\\title{" + esc(topic) + "——资源预研简报（幻灯片）}\n"
        "\\author{hydradeck}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\frame{\\titlepage}\n"
        "\\begin{frame}{关键来源}\n"
        "\\begin{itemize}\n"
        + items
        + "\n\\end{itemize}\n"
        "\\end{frame}\n"
        "\\end{document}\n"
    )


def _latexonline_compile_url(tex: str, command: str) -> str:
    q = urllib.parse.quote(tex, safe="")
    return "https://latexonline.cc/compile?text=" + q + "&command=" + command + "&force=true"


def _compile_pdf(tex: str, engine: str, backend: str) -> tuple[bytes, dict[str, list[str]]]:
    meta: dict[str, list[str]] = {"urls": [], "errors": []}
    b = backend.strip().lower()
    if b not in {"auto", "latexonline", "texlive"}:
        b = "auto"

    if b in {"auto", "latexonline"}:
        try:
            meta["urls"].append(_latexonline_compile_url(tex, command=engine))
            data = _compile_latexonline(tex, command=engine)
            _ensure_pdf_bytes(data, where="latexonline")
            return data, meta
        except Exception as e:
            meta["errors"].append("latexonline: " + str(e))
            if b == "latexonline":
                raise

    try:
        data = _compile_texlive_latexcgi(tex, engine=engine)
        _ensure_pdf_bytes(data, where="texlive")
        return data, meta
    except Exception as e:
        meta["errors"].append("texlive latexcgi: " + str(e))
        raise


def _ensure_pdf_bytes(data: bytes, where: str) -> None:
    if data.startswith(b"%PDF"):
        return
    head = data[:200].decode("utf-8", errors="replace")
    raise RuntimeError(f"{where} did not return PDF. Head: {head}")


def _compile_latexonline(tex: str, command: str) -> bytes:
    url = _latexonline_compile_url(tex, command=command)
    r = requests.get(url, timeout=120.0)
    if r.status_code >= 400:
        raise RuntimeError(f"latexonline HTTP {r.status_code}: {r.text[:2000]}")
    return r.content


def _compile_texlive_latexcgi(tex: str, engine: str) -> bytes:
    url = "https://texlive.net/cgi-bin/latexcgi"
    files = {
        "filename[]": (None, "document.tex"),
        "filecontents[]": (None, tex),
        "engine": (None, engine),
        "return": (None, "pdf"),
    }
    r = requests.post(url, files=files, timeout=120.0)
    if r.status_code >= 400:
        raise RuntimeError(f"texlive latexcgi HTTP {r.status_code}: {r.text[:2000]}")
    return r.content


def _generate_pre_tex(cfg: RunConfig, client, sources: list[Source]) -> tuple[str, str]:
    if cfg.use_mock:
        return _render_paper_tex(cfg.topic, sources), _render_slides_tex(cfg.topic, sources)

    if cfg.template.strip().lower() in {"pretty", "iclr2026"}:
        return _generate_pre_tex_pretty(cfg, client, sources)

    outline = _pre_outline(cfg.topic)
    src_json = json.dumps([asdict(s) for s in sources], ensure_ascii=False)
    feedback = ""
    last_paper = _render_paper_tex(cfg.topic, sources)
    last_slides = _render_slides_tex(cfg.topic, sources)
    for _attempt in range(max(1, cfg.pre_tex_attempts)):
        msgs = [
            ChatMessage(
                role="system",
                content=(
                    "你是严谨的 LaTeX 作者。"
                    "必须输出可用 XeLaTeX 编译的高信息密度中文内容。"
                    "不要输出 JSON。"
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "生成两个 LaTeX 文档（全部使用简体中文）：\n"
                    "(1) paper_tex：article 论文版预研报告，结构严格，信息密度高。\n"
                    "(2) slides_tex：beamer 16:9，15 分钟汇报（8-10 页）。\n\n"
                    "共同硬约束：\n"
                    "- 使用 ctex + xelatex\n"
                    "- 禁止空话；每节必须有可执行要点/表格\n"
                    "- 必须包含“参考资源”并列出全部来源 URL\n\n"
                    "paper 结构（标题可扩展但需覆盖以下要点）：\n"
                    + "\n".join(["- " + x for x in outline["paper"]])
                    + "\n\nslides 结构（每项至少一页）：\n"
                    + "\n".join(["- " + x for x in outline["slides"]])
                    + "\n\n来源 JSON：\n"
                    + src_json
                    + ("\n\n评审反馈：\n" + feedback if feedback else "")
                    + "\n\n输出格式（必须严格使用）：\n"
                    + "<<<paper.tex>>>\n<latex>\n<<<end paper.tex>>>\n"
                    + "<<<slides.tex>>>\n<latex>\n<<<end slides.tex>>>\n"
                ),
            ),
        ]
        text = client.chat_text(msgs, temperature=0.2)
        parsed = _parse_marked_tex(text)
        paper = parsed.get("paper")
        slides = parsed.get("slides")
        if not isinstance(paper, str) or not isinstance(slides, str):
            feedback = "Output must contain both <<<paper.tex>>> and <<<slides.tex>>> blocks."
            continue

        last_paper, last_slides = paper, slides
        score, fb = _score_pre_tex(paper, slides, sources)
        if not cfg.pre_tex_quality_gate or score >= cfg.pre_tex_min_score:
            return paper, slides
        feedback = fb

    return last_paper, last_slides


def _generate_pre_tex_iclr2026(
    cfg: RunConfig,
    client,
    sources: list[Source],
) -> tuple[str, str]:
    src_json = json.dumps([asdict(s) for s in sources], ensure_ascii=False)
    feedback = ""
    last_paper = ""
    last_slides = ""
    for _attempt in range(max(1, cfg.pre_tex_attempts)):
        msgs = [
            ChatMessage(
                role="system",
                content=(
                    "你撰写严谨的 ICLR 风格预研文稿。"
                    "paper 必须使用 \\usepackage{iclr2026_conference,times}。"
                    "输出必须为简体中文，不要输出 JSON。"
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "任务：撰写 (1) paper.tex（ICLR 论文风格）和 (2) slides.tex（beamer）。\n"
                    "场景：15 分钟预研汇报。\n"
                    "要求高信息密度：至少 2 张表（证据计划、风险登记）。\n"
                    "必须包含“参考资源”并列出所有来源 URL。\n\n"
                    "paper.tex 要求：\n"
                    "- Use: \\documentclass{article} and \\usepackage{iclr2026_conference,times}\n"
                    "- 包含：标题、摘要（<=150 词）\n"
                    "- 章节：目标、待验证主张、研究问题、范围/非范围\n"
                    "  证据计划（表）、来源映射、风险（表）、时间线（表）\n"
                    "  交付物、参考资源\n"
                    "- 禁止空话，每个要点必须可执行。\n\n"
                    "slides.tex 要求：\n"
                    "- 16:9 beamer, 8-10 frames, 1 idea per slide\n"
                    "- 至少 1 页证据矩阵，至少 1 页风险页\n\n"
                    "来源 JSON：\n"
                    + src_json
                    + ("\n\n反馈：\n" + feedback if feedback else "")
                    + "\n\n输出格式（必须严格）：\n"
                    + "<<<paper.tex>>>\n<latex>\n<<<end paper.tex>>>\n"
                    + "<<<slides.tex>>>\n<latex>\n<<<end slides.tex>>>\n"
                ),
            ),
        ]
        text = client.chat_text(msgs, temperature=0.2)
        parsed = _parse_marked_tex(text)
        paper = parsed.get("paper")
        slides = parsed.get("slides")
        if not isinstance(paper, str) or not isinstance(slides, str):
            feedback = "Missing marked blocks."
            continue
        last_paper, last_slides = paper, slides
        score, fb = _score_pre_tex(paper, slides, sources)
        if not cfg.pre_tex_quality_gate or score >= cfg.pre_tex_min_score:
            return paper, slides
        feedback = fb

    if last_paper and last_slides:
        return last_paper, last_slides
    return _render_paper_tex(cfg.topic, sources), _render_slides_tex(cfg.topic, sources)


def _generate_pre_tex_pretty(
    cfg: RunConfig,
    client,
    sources: list[Source],
) -> tuple[str, str]:
    src_json = json.dumps([asdict(s) for s in sources], ensure_ascii=False)
    feedback = ""
    last_paper = _render_paper_tex(cfg.topic, sources)
    last_slides = _render_slides_tex(cfg.topic, sources)

    for _attempt in range(max(1, cfg.pre_tex_attempts)):
        msgs = [
            ChatMessage(
                role="system",
                content=(
                    "你是严谨的 LaTeX 作者。"
                    "请输出可直接编译、结构完整、信息密度高的中文 .tex 文件。"
                    "不要输出 JSON。"
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "生成两个自包含 LaTeX 文件（简体中文）：\n"
                    "A) pre_paper.tex：article。\n"
                    "B) pre_slides.tex：beamer 16:9。\n\n"
                    "paper 要求：\n"
                    "- 使用 xelatex + ctex\n"
                    "- 版式整洁，信息密度高，无空话\n"
                    "- 章节至少覆盖：背景、创新、架构、能力、应用、局限、结论、参考资源\n"
                    "- 每条来源至少引用一次（\\cite{}）\n\n"
                    "slides 要求：\n"
                    "- 8-10 页，一页一核心观点\n"
                    "- 至少 1 页证据矩阵，至少 1 页风险页\n\n"
                    "来源 JSON（以此为准）：\n"
                    + src_json
                    + ("\n\n反馈：\n" + feedback if feedback else "")
                    + "\n\n输出格式（必须严格）：\n"
                    + "<<<paper.tex>>>\n<latex>\n<<<end paper.tex>>>\n"
                    + "<<<slides.tex>>>\n<latex>\n<<<end slides.tex>>>\n"
                ),
            ),
        ]
        text = client.chat_text(msgs, temperature=0.2)
        parsed = _parse_marked_tex(text)
        paper = parsed.get("paper")
        slides = parsed.get("slides")
        if not isinstance(paper, str) or not isinstance(slides, str):
            feedback = "Missing marked blocks."
            continue
        last_paper, last_slides = paper, slides

        score, fb = _score_pre_tex(paper, slides, sources)
        if "thebibliography" not in paper:
            score *= 0.75
        if not cfg.pre_tex_quality_gate or score >= cfg.pre_tex_min_score:
            return paper, slides
        feedback = fb

    return last_paper, last_slides


def _pre_outline(topic: str) -> dict[str, list[str]]:
    _ = topic
    return {
        "paper": [
            "标题",
            "1. 背景与问题定义",
            "2. 技术创新点",
            "3. 系统架构与关键机制",
            "4. 能力与性能分析",
            "5. 应用场景与价值",
            "6. 局限与风险",
            "7. 结论",
            "8. 参考资源",
        ],
        "slides": [
            "标题",
            "背景与核心问题",
            "技术创新点",
            "系统架构",
            "能力与性能",
            "应用场景",
            "局限与风险",
            "结论",
            "Q&A",
        ],
    }


def _score_pre_tex(paper: str, slides: str, sources: list[Source]) -> tuple[float, str]:
    score = 1.0
    must = [
        "背景",
        "创新",
        "架构",
        "应用",
        "局限",
        "结论",
        "参考",
    ]
    for k in must:
        if k not in paper:
            score *= 0.85
    if "\\documentclass" not in paper or "\\documentclass" not in slides:
        score *= 0.5
    if len(sources) >= 3 and paper.count("\\url{") < 3:
        score *= 0.7
    if "iclr2026_conference" in paper and "\\usepackage{iclr2026_conference" not in paper:
        score *= 0.8
    zh_chars = sum(1 for ch in (paper + slides) if "\u4e00" <= ch <= "\u9fff")
    total_chars = max(1, len(paper + slides))
    if zh_chars / total_chars < 0.15:
        score *= 0.7
    fb = "章节不足或资源映射偏弱" if score < 0.95 else "ok"
    return max(0.0, min(1.0, score)), fb


def _parse_marked_tex(text: str) -> dict[str, str]:
    def extract(name: str) -> str | None:
        start = f"<<<{name}>>>"
        end = f"<<<end {name}>>>"
        a = text.find(start)
        b = text.find(end)
        if a == -1 or b == -1 or b <= a:
            return None
        inner = text[a + len(start) : b].strip()
        inner = _strip_markdown_fences(inner).strip()
        if inner.startswith("<latex>"):
            inner = inner[len("<latex>") :].lstrip()
        return inner + "\n"

    out: dict[str, str] = {}
    paper = extract("paper.tex")
    slides = extract("slides.tex")
    if paper is not None:
        out["paper"] = paper
    if slides is not None:
        out["slides"] = slides
    return out


def _strip_markdown_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            inner = "\n".join(lines[1:-1]).strip()
            return inner + "\n"
    return s


def _dummy_pdf_bytes(label: str) -> bytes:
    content = f"Dummy PDF ({label})".encode("ascii", errors="ignore")
    return (
        b"%PDF-1.1\n"
        b"1 0 obj<<>>endobj\n"
        b"2 0 obj<< /Length 44 >>stream\n"
        b"BT /F1 12 Tf 72 720 Td ("
        + content
        + b") Tj ET\n"
        b"endstream endobj\n"
        b"3 0 obj<< /Type /Page /Parent 4 0 R /Contents 2 0 R >>endobj\n"
        b"4 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
        b"5 0 obj<< /Type /Catalog /Pages 4 0 R >>endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"trailer<< /Root 5 0 R /Size 6 >>\nstartxref\n0\n%%EOF\n"
    )
