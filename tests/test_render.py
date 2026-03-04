from __future__ import annotations

from hydradeck.core.types import ExtractedFact, Source
from hydradeck.render import (
    build_slide_frames_from_sections,
    build_slide_frames_from_report,
    enforce_slide_density,
    render_beamer_frames,
    render_paper,
    render_report_structured,
)


def test_render_paper_converts_markdown_like_body() -> None:
    sources = [Source(url="https://example.com", title="Example", snippet="snippet")]
    facts = [
        ExtractedFact(
            claim="Claim A",
            evidence="Evidence A",
            url="https://example.com",
            title="Example",
        )
    ]
    body = """## Heading
- bullet 1
- bullet 2
`inline`
```python
print('x')
```
"""

    tex = render_paper(
        topic="demo",
        outline=["背景", "创新"],
        body=body,
        facts=facts,
        sources=sources,
    )

    assert "```" not in tex
    assert "## Heading" not in tex
    assert "Heading" in tex
    assert "\\begin{itemize}" in tex
    assert "bullet 1" in tex
    assert "\\section*{1. Introduction and Background}" in tex


def test_render_templates_use_facts_not_generic_filler() -> None:
    sources = [Source(url="https://example.com", title="Example", snippet="snippet")]
    facts = [
        ExtractedFact(
            claim="RynnBrain released checkpoints on 2026-02-09",
            evidence="project timeline from official repo",
            url="https://example.com",
            title="Example",
        ),
        ExtractedFact(
            claim="Model introduces interleaved reasoning with spatial grounding",
            evidence="technical report description",
            url="https://example.com",
            title="Example",
        ),
    ]
    paper = render_paper(
        topic="demo",
        outline=["背景", "创新", "架构"],
        body="结论段落 [1]",
        facts=facts,
        sources=sources,
    )
    section_blocks = [
        {"name": "背景", "latex": facts[0].claim},
        {"name": "创新", "latex": facts[1].claim},
    ]
    frames = build_slide_frames_from_sections(section_blocks, language="en")
    slides = render_beamer_frames("demo", frames, language="en")

    assert "released checkpoints" in paper
    assert "interleaved reasoning" in paper
    assert "released checkpoints" in slides
    assert "interleaved reasoning" in slides
    section_one = paper.split("\\section*{1. Introduction and Background}", 1)[1]
    section_one = section_one.split("\\section*{2. Logical Outline}", 1)[0]
    assert "\\begin{itemize}" not in section_one


def test_render_beamer_from_report_derives_outline() -> None:
    paper = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\section*{Executive Summary}\nAlpha beta gamma.\n\n"
        "\\section*{Methodology}\nMethod details.\n\n"
        "\\section*{Results}\nResult details.\n"
        "\\end{document}\n"
    )
    frames = build_slide_frames_from_report(paper, language="en")
    slides = render_beamer_frames("demo", frames, language="en")
    assert "\\begin{frame}{Agenda}" in slides
    assert "Executive Summary" in slides
    assert "Methodology" in slides


def test_render_beamer_frames_limits_density() -> None:
    report = (
        "\\documentclass{article}\\begin{document}"
        "\\section*{Overview} A long sentence about architecture and implementation details repeated."
        " Another long sentence about evaluation metrics and reproducibility details."
        "\\section*{Results} Multiple findings with evidence and quantitative metrics."
        "\\end{document}"
    )
    frames = build_slide_frames_from_report(report, language="en")
    assert len(frames) >= 2
    tex = render_beamer_frames("demo", frames, language="en")
    assert "\\begin{frame}{Agenda}" in tex
    assert "\\begin{itemize}" in tex


def test_render_report_structured_zh_uses_ctex() -> None:
    section_blocks = [
        {"name": "方法", "latex": "本节给出方法细节与参数说明。"},
        {"name": "结果", "latex": "本节给出结果与证据。"},
    ]
    tex = render_report_structured("中文研究报告", section_blocks, language="zh")
    assert "\\documentclass[11pt]{ctexart}" in tex
    assert "本研究报告聚焦可追溯证据" not in tex
    assert "建议定期刷新证据并进行复跑验证" not in tex
    assert "\\section*{方法}" in tex


def test_render_beamer_frames_zh_uses_ctexbeamer() -> None:
    frames = [
        build_slide_frames_from_report(
            "\\documentclass{ctexart}\\begin{document}\\section*{结果}关键结果一。关键结果二。\\end{document}",
            language="zh",
        )[0]
    ]
    tex = render_beamer_frames("中文主题", frames, language="zh")
    assert "\\documentclass[aspectratio=169]{ctexbeamer}" in tex
    assert "\\begin{frame}{目录}" in tex


def test_build_slide_frames_from_sections_splits_long_section() -> None:
    section_blocks = [
        {
            "name": "Results",
            "latex": (
                "The first finding shows strong improvement in consistency and precision. "
                "The second finding shows stronger robustness under distribution shift. "
                "The third finding indicates cost-performance improvement. "
                "The fourth finding confirms stability across runs. "
                "The fifth finding highlights limitations and guardrails."
            ),
        }
    ]
    frames = build_slide_frames_from_sections(section_blocks, language="en")
    assert len(frames) >= 2


def test_render_report_structured_removes_bracket_refs() -> None:
    section_blocks = [
        {"name": "Evidence", "latex": "Claim [1] with support [2] and \\cite{src1}."}
    ]
    tex = render_report_structured("demo", section_blocks, language="en")
    assert "[1]" not in tex
    assert "[2]" not in tex
    assert "\\cite{" not in tex


def test_enforce_slide_density_splits_large_bullet_groups() -> None:
    frames = [
        {
            "title": "Results",
            "bullets": [
                "point one with enough words to be valid",
                "point two with enough words to be valid",
                "point three with enough words to be valid",
                "point four with enough words to be valid",
                "point five with enough words to be valid",
            ],
        }
    ]
    from hydradeck.render import SlideFrame

    fr = [SlideFrame(title=x["title"], bullets=x["bullets"]) for x in frames]
    out = enforce_slide_density(fr, language="en", max_bullets_per_frame=4)
    assert len(out) == 2
    assert out[0].title == "Results"
    assert "(cont.)" in out[1].title
