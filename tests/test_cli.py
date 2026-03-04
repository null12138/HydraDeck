from __future__ import annotations

import pytest

from hydradeck import cli
from hydradeck.core.types import RunConfig


def test_run_command_accepts_snapshot_total_timeout_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    def fake_run(cfg: RunConfig) -> object:
        captured["snapshot_total_timeout_s"] = cfg.snapshot_total_timeout_s
        return object()

    monkeypatch.setattr(cli, "run", fake_run)

    code = cli.main(
        [
            "run",
            "--topic",
            "t",
            "--out",
            "out.zip",
            "--base-url",
            "https://example.invalid",
            "--model",
            "mock",
            "--mock",
        ]
    )

    assert code == 0
    assert captured["snapshot_total_timeout_s"] == 60.0


def test_run_command_passes_request_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float] = {}

    def fake_run(cfg: RunConfig) -> object:
        captured["request_budget_s"] = cfg.request_budget_s
        return object()

    monkeypatch.setattr(cli, "run", fake_run)

    code = cli.main(
        [
            "run",
            "--topic",
            "t",
            "--out",
            "out.zip",
            "--base-url",
            "https://example.invalid",
            "--model",
            "mock",
            "--request-budget",
            "90",
            "--mock",
        ]
    )

    assert code == 0
    assert captured["request_budget_s"] == 90.0
