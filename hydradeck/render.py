from __future__ import annotations

import re
from dataclasses import dataclass

from hydradeck.core.types import ExtractedFact, Source

_LATEX_SPECIALS: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "#": r"\#",
    "$": r"\$",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
}


def latex_escape(s: str) -> str:
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in s)


def _bib_key(i: int) -> str:
    return f"src{i}"


def _bib_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def render_bibtex(sources: list[Source]) -> str:
    lines: list[str] = []
    for i, s in enumerate(sources, start=1):
        key = _bib_key(i)
        lines.append(f"@misc{{{key},")
        lines.append(f"  title = {{{_bib_escape(s.title)}}},")
        lines.append(f"  howpublished = {{\\url{{{_bib_escape(s.url)}}}}},")
        lines.append("  note = {Accessed: 2026-03-04},")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _replace_numeric_citations(text: str, max_n: int) -> str:
    def repl(m: re.Match[str]) -> str:
        num = int(m.group(1))
        if 1 <= num <= max_n:
            return f"\\cite{{{_bib_key(num)}}}"
        return m.group(0)

    return re.sub(r"\[(\d{1,3})\]", repl, text)


def _markdown_to_latex_paragraphs(md: str, max_n: int) -> str:
    text = md.strip()
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = _replace_numeric_citations(text, max_n=max_n)
    text = latex_escape(text)
    text = re.sub(r"\\textbackslash\{\}cite\\\{(src\d+)\\\}", r"\\cite{\1}", text)
    text = text.replace("\n\n", "\n\\par\n")
    return text


def render_paper(
    topic: str,
    outline: list[str],
    body: str,
    facts: list[ExtractedFact],
    sources: list[Source],
) -> str:
    topic_e = latex_escape(topic)
    url_to_key = {s.url: _bib_key(i) for i, s in enumerate(sources, start=1)}

    outline_items = "\n".join([f"\\item {latex_escape(x)}" for x in outline[:10]])
    fact_sentences: list[str] = []
    for f in facts[:18]:
        key = url_to_key.get(f.url)
        cite = f"\\cite{{{key}}}" if key else ""
        sentence = latex_escape(f.claim.strip())
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        fact_sentences.append(sentence + (cite if cite else ""))
    facts_paragraph = (
        " ".join(fact_sentences)
        if fact_sentences
        else "No extracted facts available."
    )

    body_latex = _markdown_to_latex_paragraphs(body, max_n=len(sources))

    return (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage{geometry}\n"
        "\\usepackage{hyperref}\n"
        "\\usepackage{url}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{longtable}\n"
        "\\geometry{margin=1in}\n"
        "\\hypersetup{colorlinks=true,linkcolor=black,citecolor=blue,urlcolor=blue}\n"
        f"\\title{{{topic_e}}}\n"
        "\\author{hydradeck}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\begin{abstract}\n"
        "This report presents a structured analysis with explicit traceability to sources.\n"
        "\\end{abstract}\n\n"
        "\\section*{1. Introduction and Background}\n"
        + facts_paragraph
        + "\n\n"
        "\\section*{2. Logical Outline}\n"
        "\\begin{itemize}\n"
        + outline_items
        + "\n\\end{itemize}\n\n"
        "\\section*{3. Evidence and Key Findings}\n"
        + body_latex
        + "\n\n"
        "\\section*{4. Limitations and Discussion}\n"
        "The analysis is bounded by available public evidence and may evolve as sources update.\n\n"
        "\\section*{5. Conclusion}\n"
        "Conclusions are presented in a source-traceable form and should be interpreted with the\n"
        "reported assumptions and constraints.\n\n"
        "\\bibliographystyle{plain}\n"
        "\\bibliography{refs}\n"
        "\\end{document}\n"
    )


def render_report_structured(
    topic: str,
    section_blocks: list[dict[str, str]],
    language: str = "en",
) -> str:
    lang = language.lower()
    topic_e = latex_escape(topic)

    if lang == "zh":
        preamble = (
            "\\documentclass[11pt]{ctexart}\n"
            "\\usepackage[a4paper,margin=1in]{geometry}\n"
            "\\usepackage{hyperref}\n"
            "\\usepackage{url}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{longtable}\n"
            "\\hypersetup{colorlinks=true,linkcolor=black,citecolor=blue,urlcolor=blue}\n"
            f"\\title{{{topic_e}}}\n"
            "\\author{hydradeck}\n"
            "\\date{\\today}\n"
            "\\begin{document}\n"
            "\\maketitle\n"
        )
    else:
        preamble = (
            "\\documentclass[11pt]{article}\n"
            "\\usepackage{geometry}\n"
            "\\usepackage{hyperref}\n"
            "\\usepackage{url}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{longtable}\n"
            "\\geometry{margin=1in}\n"
            "\\hypersetup{colorlinks=true,linkcolor=black,citecolor=blue,urlcolor=blue}\n"
            f"\\title{{{topic_e}}}\n"
            "\\author{hydradeck}\n"
            "\\date{\\today}\n"
            "\\begin{document}\n"
            "\\maketitle\n"
        )

    content_parts: list[str] = []
    for block in section_blocks[:10]:
        title = latex_escape(str(block.get("name", "Section")).strip() or "Section")
        latex_body = str(block.get("latex", "")).strip()
        latex_body = re.sub(r"\\section\*?\{[^}]*\}", "", latex_body)
        latex_body = re.sub(r"\\subsection\*?\{[^}]*\}", "", latex_body)
        latex_body = re.sub(r"\\cite\{[^}]*\}", "", latex_body)
        latex_body = re.sub(r"\[(\d{1,3})\]", "", latex_body)
        if not latex_body:
            continue
        content_parts.append(f"\\section*{{{title}}}\n{latex_body}\n")

    return preamble + "\n".join(content_parts) + "\n\\end{document}\n"


@dataclass
class SlideFrame:
    title: str
    bullets: list[str]
    note: str = ""


def render_beamer(topic: str, outline: list[str], bullets: list[str]) -> str:
    section_blocks = [{"name": t, "latex": b} for t, b in zip(outline, bullets)]
    if not section_blocks:
        section_blocks = [{"name": "Summary", "latex": "Key findings and implications."}]
    frames = build_slide_frames_from_sections(section_blocks, language="en")
    frames = enforce_slide_density(frames, language="en")
    return render_beamer_frames(topic, frames, language="en")


def render_beamer_from_report(topic: str, report_tex: str) -> str:
    frames = build_slide_frames_from_report(report_tex, language="en")
    frames = enforce_slide_density(frames, language="en")
    return render_beamer_frames(topic, frames, language="en")


def _split_paragraph_to_bullets(text: str, language: str) -> list[str]:
    lang = language.lower()
    if lang == "zh":
        parts = [x.strip() for x in re.split(r"[。！？]\s*", text) if x.strip()]
        out: list[str] = []
        for p in parts:
            if len(p) < 6:
                continue
            out.append(_trim_chars(_clean_text_for_slide(p), 28))
        return out

    parts = [x.strip() for x in re.split(r"[.!?]\s+", text) if x.strip()]
    out2: list[str] = []
    for p in parts:
        clean = _clean_text_for_slide(p)
        if len(clean) < 14:
            continue
        out2.append(_trim_words(clean, 14))
    return out2


def build_slide_frames_from_sections(
    section_blocks: list[dict[str, str]],
    language: str = "en",
) -> list[SlideFrame]:
    lang = language.lower()
    frames: list[SlideFrame] = []
    for block in section_blocks[:8]:
        title = str(block.get("name", "Section")).strip() or ("章节" if lang == "zh" else "Section")
        body = str(block.get("latex", ""))
        body = re.sub(r"\\section\*?\{[^}]*\}", "", body)
        body = re.sub(r"\\subsection\*?\{[^}]*\}", "", body)
        body = re.sub(r"\\cite\{[^}]*\}", "", body)
        body = re.sub(r"\[(\d{1,3})\]", "", body)
        bullets = _split_paragraph_to_bullets(body, lang)
        if not bullets:
            continue

        chunk = 4 if lang == "zh" else 4
        for i in range(0, len(bullets), chunk):
            part = bullets[i : i + chunk]
            if not part:
                continue
            if i == 0:
                frame_title = title
            else:
                frame_title = f"{title}（续）" if lang == "zh" else f"{title} (cont.)"
            frames.append(SlideFrame(title=frame_title, bullets=part))

    if not frames:
        raise RuntimeError("insufficient readable section content for slides")
    return frames


def enforce_slide_density(
    frames: list[SlideFrame],
    language: str = "en",
    max_bullets_per_frame: int = 4,
    max_chars_per_bullet_zh: int = 28,
    max_words_per_bullet_en: int = 14,
) -> list[SlideFrame]:
    lang = language.lower()
    out: list[SlideFrame] = []

    for fr in frames:
        normalized: list[str] = []
        for b in fr.bullets:
            clean = _clean_text_for_slide(b)
            if not clean:
                continue
            if lang == "zh":
                clean = _trim_chars(clean, max_chars_per_bullet_zh)
            else:
                clean = _trim_words(clean, max_words_per_bullet_en)
            if clean:
                normalized.append(clean)

        if not normalized:
            continue

        for i in range(0, len(normalized), max_bullets_per_frame):
            chunk = normalized[i : i + max_bullets_per_frame]
            if not chunk:
                continue
            if i == 0:
                title = fr.title
            else:
                title = f"{fr.title}（续）" if lang == "zh" else f"{fr.title} (cont.)"
            out.append(SlideFrame(title=title, bullets=chunk, note=fr.note))

    if not out:
        raise RuntimeError("slide density guard removed all frames")
    return out


def _trim_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,.;") + "..."


def _trim_chars(text: str, max_chars: int) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip("，。,. ") + "…"


def _clean_text_for_slide(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"\*(.*?)\*", r"\1", t)
    return t


def build_slide_frames_from_report(report_tex: str, language: str = "en") -> list[SlideFrame]:
    lang = language.lower()
    sections = re.split(r"\\section\*\{([^}]+)\}", report_tex)
    parsed: list[tuple[str, str]] = []
    if len(sections) >= 3:
        for i in range(1, len(sections), 2):
            title = sections[i].strip()
            body = sections[i + 1] if i + 1 < len(sections) else ""
            parsed.append((title, body))

    if not parsed:
        raise RuntimeError("cannot derive slide frames from report structure")

    frames: list[SlideFrame] = []
    for title, body in parsed[:8]:
        plain = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", body)
        chunks = [x.strip() for x in re.split(r"[。.!?]\s+", plain) if x.strip()]
        bullets: list[str] = []
        for c in chunks:
            clean = _clean_text_for_slide(c)
            if not clean:
                continue
            if lang == "zh":
                if len(clean) < 8:
                    continue
                bullets.append(_trim_chars(clean, 30))
            else:
                if len(clean) < 12:
                    continue
                bullets.append(_trim_words(clean, 16))
            if len(bullets) >= 5:
                break
        if not bullets:
            raise RuntimeError(f"insufficient bullet content for slide '{title}'")
        frames.append(SlideFrame(title=title, bullets=bullets))

    return frames


def render_beamer_frames(topic: str, frames: list[SlideFrame], language: str = "en") -> str:
    lang = language.lower()
    topic_e = latex_escape(topic)
    agenda_label = "目录" if lang == "zh" else "Agenda"
    summary_title = "总结" if lang == "zh" else "Summary"

    agenda_items = "\n".join([f"\\item {latex_escape(f.title)}" for f in frames[:8]])

    frame_blocks: list[str] = []
    for fr in frames[:10]:
        b = "\n".join([f"\\item {latex_escape(x)}" for x in fr.bullets[:5]])
        frame_blocks.append(
            "\\begin{frame}[t]{"
            + latex_escape(fr.title)
            + "}\n"
            + "\\begin{itemize}\n"
            + b
            + "\n\\end{itemize}\n"
            + (f"\\vspace{{0.6em}}\\footnotesize {latex_escape(fr.note)}\n" if fr.note else "")
            + "\\end{frame}\n"
        )

    summary_bullets: list[str] = []
    for fr in frames[:5]:
        if fr.bullets:
            summary_bullets.append(fr.bullets[0])
    if not summary_bullets:
        summary_bullets = ["关键要点见前页。" if lang == "zh" else "Key points are summarized in previous slides."]
    summary_items = "\n".join([f"\\item {latex_escape(x)}" for x in summary_bullets])

    if lang == "zh":
        return (
            "\\documentclass[aspectratio=169]{ctexbeamer}\n"
            "\\usetheme{Madrid}\n"
            "\\usefonttheme{professionalfonts}\n"
            "\\setbeamertemplate{navigation symbols}{}\n"
            "\\usepackage{hyperref}\n"
            "\\usepackage{booktabs}\n"
            "\\definecolor{AccentBlue}{HTML}{1F4E79}\n"
            "\\setbeamercolor{title}{fg=AccentBlue}\n"
            "\\setbeamercolor{frametitle}{fg=AccentBlue}\n"
            "\\setbeamerfont{title}{series=\\bfseries,size=\\Large}\n"
            "\\setbeamerfont{frametitle}{series=\\bfseries,size=\\large}\n"
            f"\\title{{{topic_e}}}\n"
            "\\author{hydradeck}\n"
            "\\date{\\today}\n"
            "\\begin{document}\n"
            "\\frame{\\titlepage}\n"
            "\\begin{frame}{"
            + latex_escape(agenda_label)
            + "}\n"
            "\\begin{itemize}\n"
            + agenda_items
            + "\n\\end{itemize}\n"
            "\\end{frame}\n"
            + "".join(frame_blocks)
            + "\\begin{frame}{"
            + latex_escape(summary_title)
            + "}\n"
            + "\\begin{itemize}\n"
            + summary_items
            + "\n\\end{itemize}\n"
            + "\\end{frame}\n"
            + "\\end{document}\n"
        )

    return (
        "\\documentclass[aspectratio=169]{beamer}\n"
        "\\usetheme{metropolis}\n"
        "\\usefonttheme{professionalfonts}\n"
        "\\setbeamertemplate{navigation symbols}{}\n"
        "\\usepackage{hyperref}\n"
        "\\usepackage{booktabs}\n"
        "\\definecolor{AccentBlue}{HTML}{1F4E79}\n"
        "\\setbeamercolor{title}{fg=AccentBlue}\n"
        "\\setbeamercolor{frametitle}{fg=AccentBlue}\n"
        "\\setbeamerfont{title}{series=\\bfseries,size=\\Large}\n"
        "\\setbeamerfont{frametitle}{series=\\bfseries,size=\\large}\n"
        f"\\title{{{topic_e}}}\n"
        "\\author{hydradeck}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\frame{\\titlepage}\n"
        "\\begin{frame}{"
        + latex_escape(agenda_label)
        + "}\n"
        "\\begin{itemize}\n"
        + agenda_items
        + "\n\\end{itemize}\n"
        "\\end{frame}\n"
        + "".join(frame_blocks)
        + "\\begin{frame}{"
        + latex_escape(summary_title)
        + "}\n"
        + "\\begin{itemize}\n"
        + summary_items
        + "\n\\end{itemize}\n"
        + "\\end{frame}\n"
        + "\\end{document}\n"
    )
