from __future__ import annotations

import zipfile
from pathlib import Path

from hydradeck.core.types import RunConfig
from hydradeck.resources_pack import build_resources_pack


def test_resources_pack_mock(tmp_path: Path) -> None:
    out_zip = tmp_path / "res.zip"
    cfg = RunConfig(
        topic="RynnBrain",
        out=out_zip,
        base_url="https://example.invalid",
        api_key="",
        model="mock",
        use_mock=True,
        verbose=False,
        progress=False,
        llm_timeout_s=5.0,
        max_total_runtime_s=5.0,
        request_budget_s=2.0,
        snapshot_timeout_s=1.0,
        keep_stage=False,
        max_sources=3,
        module_sources=2,
    )
    build_resources_pack(cfg)
    assert out_zip.exists()
    with zipfile.ZipFile(out_zip, "r") as z:
        names = set(z.namelist())
        pre_paper = z.read("pre_paper.tex").decode("utf-8")
        pre_slides = z.read("pre_slides.tex").decode("utf-8")
    assert "resources/sources.json" in names
    assert "resources/snapshots.json" in names
    assert "research.json" in names
    assert "pre_paper.tex" in names
    assert "pre_slides.tex" in names
    assert "pdf/pre_paper.pdf" in names
    assert "pdf/pre_slides.pdf" in names
    assert "来源清单" in pre_paper
    assert "关键来源" in pre_slides
