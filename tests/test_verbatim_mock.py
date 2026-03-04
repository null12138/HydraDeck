from __future__ import annotations

import zipfile
from pathlib import Path

from hydradeck.core.types import RunConfig
from hydradeck.pipeline import run


def test_mock_run_verbatim_creates_outputs(tmp_path: Path) -> None:
    out_zip = tmp_path / "verbatim.zip"
    cfg = RunConfig(
        topic="RynnBrain",
        out=out_zip,
        base_url="https://example.invalid",
        api_key="",
        model="mock",
        use_mock=True,
        verbose=False,
        iterations=1,
        max_sources=3,
        verbatim=True,
        archive_prompts=True,
        archive_snapshots=False,
        quality_gate=True,
        min_quality_score=0.85,
        max_quality_attempts=2,
    )
    run(cfg)
    with zipfile.ZipFile(out_zip, "r") as z:
        names = set(z.namelist())
    for required in [
        "pre_report.md",
        "report.md",
        "speech.md",
        "paper.tex",
        "slides.tex",
        "refs.bib",
        "research.json",
        "resources/sources.json",
        "prompts.jsonl",
    ]:
        assert required in names
