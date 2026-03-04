from __future__ import annotations

import zipfile
from pathlib import Path

from hydradeck.core.types import RunConfig
from hydradeck.pipeline import run


def test_mock_run_creates_zip(tmp_path: Path) -> None:
    out_zip = tmp_path / "demo.zip"
    cfg = RunConfig(
        topic="test topic",
        out=out_zip,
        base_url="https://example.invalid",
        api_key="",
        model="mock",
        use_mock=True,
        verbose=False,
        iterations=2,
        max_sources=3,
        archive_snapshots=False,
        auto=True,
        auto_queries=True,
        auto_models=True,
    )
    run(cfg)
    assert out_zip.exists()
    with zipfile.ZipFile(out_zip, "r") as z:
        names = set(z.namelist())
        compile_sh = z.read("compile.sh").decode("utf-8")
        paper_tex = z.read("paper.tex").decode("utf-8")
        slides_tex = z.read("slides.tex").decode("utf-8")
    for required in [
        "pre_report.md",
        "report.md",
        "speech.md",
        "paper.tex",
        "slides.tex",
        "refs.bib",
        "research.json",
        "compile.sh",
        "Makefile",
        "resources/sources.json",
    ]:
        assert required in names

    assert "xelatex -interaction=nonstopmode paper.tex" in compile_sh
    assert "xelatex -interaction=nonstopmode slides.tex" in compile_sh
    assert "\\section*{3. Evidence and Key Findings}" in paper_tex
    assert "\\section*{1. Introduction and Background}" in paper_tex
    assert "\\begin{frame}{Agenda}" in slides_tex
    assert "\\usetheme{metropolis}" in slides_tex
    assert "```" not in paper_tex
    assert "```" not in slides_tex
    assert "## " not in paper_tex
    assert "## " not in slides_tex
