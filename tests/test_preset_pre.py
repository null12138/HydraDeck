from __future__ import annotations

import zipfile
from pathlib import Path


def test_preset_rynnbrain_zip(tmp_path: Path) -> None:
    from hydradeck.presets.rynnbrain import generate

    out_zip = tmp_path / "rynnbrain_pre.zip"
    generate(out=out_zip, keep_stage=False, fetch=False)
    assert out_zip.exists()
    with zipfile.ZipFile(out_zip, "r") as z:
        names = set(z.namelist())
    assert "pre_report.md" in names
    assert "research.json" in names
    assert "resources/sources.json" in names
