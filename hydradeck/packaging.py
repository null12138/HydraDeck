from __future__ import annotations

import shutil
import zipfile
from collections.abc import Iterable
from pathlib import Path


def is_zip_path(p: Path) -> bool:
    return p.suffix.lower() == ".zip"


def stage_dir_for_out(out: Path) -> Path:
    if is_zip_path(out):
        return out.with_suffix("")
    return out


def create_zip(zip_path: Path, src_dir: Path, members: Iterable[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in members:
            rel = p.relative_to(src_dir)
            z.write(str(p), arcname=str(rel))


def finalize_output(out: Path, stage_dir: Path, keep_stage: bool = False) -> None:
    if not is_zip_path(out):
        return
    files = [p for p in stage_dir.rglob("*") if p.is_file()]
    create_zip(out, stage_dir, files)
    if not keep_stage:
        shutil.rmtree(stage_dir, ignore_errors=True)
