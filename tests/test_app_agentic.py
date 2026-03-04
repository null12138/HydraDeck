from __future__ import annotations

from pathlib import Path

import app


def test_agentic_pipeline_mock_renders_online_pdfs(monkeypatch) -> None:
    def fake_compile(tex_source: str, output_name: str) -> str:
        p = Path("/tmp") / output_name
        p.write_bytes(b"%PDF-1.5\n%mock\n")
        return str(p)

    monkeypatch.setattr(app, "_compile_latex_online", fake_compile)

    (
        status,
        progress_log,
        _scope_json,
        section_plan_json,
        paper_tex,
        slides_tex,
        rendered_pdfs,
        paper_pdf,
        slides_pdf,
    ) = (
        app._run_agentic_pipeline(
            topic="Agentic flow test",
            model="grok-3-mini",
            base_url="https://api.example.com",
            api_key="",
            request_budget=20,
            use_mock=True,
        )
    )

    assert "done" in status.lower()
    assert "ScopeScout" in progress_log
    assert "sections" in section_plan_json
    assert "documentclass" in paper_tex
    assert "documentclass" in slides_tex
    paths = [x.strip() for x in rendered_pdfs.splitlines() if x.strip()]
    assert len(paths) == 2
    for p in paths:
        assert Path(p).exists()
    assert Path(str(paper_pdf)).exists()
    assert Path(str(slides_pdf)).exists()


def test_agentic_stream_emits_progress_and_pdf_paths(monkeypatch) -> None:
    def fake_compile(tex_source: str, output_name: str) -> str:
        p = Path("/tmp") / output_name
        p.write_bytes(b"%PDF-1.5\n%mock\n")
        return str(p)

    monkeypatch.setattr(app, "_compile_latex_online", fake_compile)

    chunks = list(
        app._run_agentic_pipeline_stream(
            topic="Agentic stream test",
            model="grok-3-mini",
            base_url="https://api.example.com",
            api_key="",
            request_budget=20,
            use_mock=True,
        )
    )
    assert len(chunks) >= 3
    assert chunks[0][-1] == 5
    assert chunks[1][-1] == 30
    assert chunks[-1][-1] == 100
    assert "done" in str(chunks[-1][0]).lower()
    assert Path(str(chunks[-1][7])).exists()
    assert Path(str(chunks[-1][8])).exists()
