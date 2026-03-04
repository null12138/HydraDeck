from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydradeck.config import (
    UserConfig,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
    resolve_pdf_compiler,
    resolve_template,
    save_config,
)
from hydradeck.core.types import RunConfig
from hydradeck.pipeline import run
from hydradeck.resources_pack import build_resources_pack


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hydradeck")
    sub = p.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Run Grok deep research pipeline")
    runp.add_argument("--topic", required=True, help="Research topic")
    runp.add_argument("--out", required=True, help="Output directory or .zip")
    runp.add_argument("--iterations", type=int, default=3, help="Persona iteration rounds")
    runp.add_argument("--max-sources", type=int, default=10, help="Max sources to include")
    runp.add_argument(
        "--min-words",
        type=int,
        default=12000,
        help="Target minimum words (guidance to model; markdown is primary)",
    )
    runp.add_argument("--base-url", default=None, help="API base URL")
    runp.add_argument("--model", default=None, help="Model name")
    runp.add_argument(
        "--keep-stage",
        action="store_true",
        help="If --out is a .zip, keep the staging directory on disk",
    )
    runp.add_argument(
        "--seed-url",
        action="append",
        default=None,
        help="Seed URL to include as source (can be repeated)",
    )
    runp.add_argument("--llm-timeout", type=float, default=180.0, help="LLM timeout seconds")
    runp.add_argument("--mock", action="store_true", help="Use deterministic mock (no network)")
    runp.add_argument("--verbose", action="store_true", help="Verbose logging")
    runp.add_argument(
        "--heartbeat",
        action="store_true",
        help="Emit periodic heartbeat during long network calls",
    )
    runp.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar for generation stages",
    )
    runp.add_argument(
        "--request-budget",
        type=float,
        default=20.0,
        help="Per-request timeout budget (seconds)",
    )
    runp.add_argument(
        "--verbatim",
        action="store_true",
        help="Write model-produced artifacts verbatim (no rendering/rewriting)",
    )
    runp.add_argument(
        "--no-archive-prompts",
        action="store_true",
        help="Do not archive prompts/requests in the output package",
    )
    runp.add_argument(
        "--quality-gate",
        action="store_true",
        help="Require passing third-party score before writing outputs",
    )
    runp.add_argument(
        "--min-quality",
        type=float,
        default=0.85,
        help="Minimum quality score (0-1)",
    )
    runp.add_argument(
        "--quality-attempts",
        type=int,
        default=3,
        help="Max regeneration attempts to meet quality gate",
    )
    runp.add_argument(
        "--archive-snapshots",
        action="store_true",
        help="Fetch and archive source page snapshots into resources/snapshots",
    )
    runp.add_argument(
        "--snapshot-timeout",
        type=float,
        default=25.0,
        help="Per-URL snapshot fetch timeout (seconds)",
    )
    runp.add_argument(
        "--snapshot-total-timeout",
        type=float,
        default=60.0,
        help="Total time budget for all snapshots (seconds)",
    )

    prep = sub.add_parser(
        "pre",
        help="Generate a preset pre-research package (no API key required)",
    )
    prep.add_argument("--preset", required=True, help="Preset name (e.g. rynnbrain)")
    prep.add_argument("--out", required=True, help="Output directory or .zip")
    prep.add_argument(
        "--keep-stage",
        action="store_true",
        help="Keep staging directory when output is .zip",
    )
    prep.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not fetch and archive web snapshots",
    )

    models_p = sub.add_parser("models", help="List available models")
    models_p.add_argument(
        "--base-url",
        default=None,
        help="API base URL",
    )

    auto_p = sub.add_parser(
        "auto",
        help="Run autonomous deep research (verbatim + prompts + snapshots)",
    )
    auto_p.add_argument("--topic", required=True, help="Research topic")
    auto_p.add_argument("--out", required=True, help="Output directory or .zip")
    auto_p.add_argument(
        "--base-url",
        default=None,
        help="API base URL",
    )
    auto_p.add_argument(
        "--model",
        default=None,
        help="Fallback model name",
    )
    auto_p.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Persona iteration rounds",
    )
    auto_p.add_argument(
        "--max-sources",
        type=int,
        default=12,
        help="Max sources to include",
    )
    auto_p.add_argument(
        "--module-sources",
        type=int,
        default=5,
        help="Sources per query module",
    )
    auto_p.add_argument(
        "--query-count",
        type=int,
        default=8,
        help="Number of queries to generate (high recall)",
    )
    auto_p.add_argument(
        "--max-query-modules",
        type=int,
        default=2,
        help="Max query modules to expand into sources",
    )
    auto_p.add_argument(
        "--sources-attempts",
        type=int,
        default=3,
        help="Max attempts to obtain sources (must be <=3)",
    )
    auto_p.add_argument(
        "--facts-max-pages",
        type=int,
        default=6,
        help="Max pages to pass into facts extraction",
    )
    auto_p.add_argument(
        "--facts-max-chars",
        type=int,
        default=8000,
        help="Max chars per page passed into facts extraction",
    )
    auto_p.add_argument(
        "--facts-target",
        type=int,
        default=30,
        help="Approximate number of facts to extract",
    )
    auto_p.add_argument(
        "--judge-max-chars",
        type=int,
        default=12000,
        help="Max chars per artifact passed into judge",
    )
    auto_p.add_argument(
        "--max-runtime",
        type=float,
        default=240.0,
        help="Max total runtime seconds before aborting",
    )
    auto_p.add_argument(
        "--llm-timeout",
        type=float,
        default=180.0,
        help="LLM timeout seconds",
    )
    auto_p.add_argument(
        "--snapshot-timeout",
        type=float,
        default=25.0,
        help="Per-URL snapshot fetch timeout (seconds)",
    )
    auto_p.add_argument("--mock", action="store_true", help="Use deterministic mock")
    auto_p.add_argument("--verbose", action="store_true", help="Verbose logging")
    auto_p.add_argument(
        "--heartbeat",
        action="store_true",
        help="Emit periodic heartbeat during long network calls",
    )
    auto_p.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar for generation stages",
    )
    auto_p.add_argument(
        "--request-budget",
        type=float,
        default=20.0,
        help="Per-request timeout budget (seconds)",
    )
    auto_p.add_argument(
        "--min-quality",
        type=float,
        default=0.85,
        help="Minimum quality score (0-1)",
    )
    auto_p.add_argument(
        "--quality-attempts",
        type=int,
        default=3,
        help="Max regeneration attempts to meet quality gate",
    )

    cfg_p = sub.add_parser("config", help="Persist local config (base_url/model/api_key)")
    cfg_p.add_argument("--base-url", default=None, help="API base URL")
    cfg_p.add_argument("--model", default=None, help="Default model")
    cfg_p.add_argument("--api-key", default=None, help="API key (stored locally)")
    cfg_p.add_argument(
        "--pdf-compiler",
        default=None,
        help="PDF compiler backend: latexonline or texlive",
    )
    cfg_p.add_argument(
        "--template",
        default=None,
        help="Template: iclr2026 or plain",
    )

    res_p = sub.add_parser("resources", help="One-click resources pack (no seed required)")
    res_p.add_argument("--topic", required=True, help="Research topic")
    res_p.add_argument("--out", required=True, help="Output directory or .zip")
    res_p.add_argument(
        "--base-url",
        default=None,
        help="API base URL",
    )
    res_p.add_argument(
        "--model",
        default=None,
        help="Model name",
    )
    res_p.add_argument(
        "--pdf-compiler",
        default=resolve_pdf_compiler("auto"),
        help="PDF compiler: auto|latexonline|texlive",
    )
    res_p.add_argument(
        "--template",
        default=resolve_template("pretty"),
        help="Template: pretty|plain",
    )
    res_p.add_argument("--max-sources", type=int, default=8, help="Max sources")
    res_p.add_argument("--module-sources", type=int, default=3, help="Sources per module")
    res_p.add_argument("--llm-timeout", type=float, default=35.0, help="LLM timeout")
    res_p.add_argument("--snapshot-timeout", type=float, default=10.0, help="Snapshot timeout")
    res_p.add_argument(
        "--snapshot-total-timeout",
        type=float,
        default=60.0,
        help="Total time budget for all snapshots",
    )
    res_p.add_argument("--max-runtime", type=float, default=180.0, help="Max runtime")
    res_p.add_argument("--request-budget", type=float, default=15.0, help="Per-request budget")
    res_p.add_argument("--keep-stage", action="store_true", help="Keep staging directory")
    res_p.add_argument("--heartbeat", action="store_true", help="Heartbeat")
    res_p.add_argument("--progress", action="store_true", help="Progress bar")

    wiz_p = sub.add_parser("wizard", help="Guided research (interactive)")
    wiz_p.add_argument("--out", required=False, default=None, help="Output directory or .zip")
    return p


def _prompt(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    v = input(prompt + suffix + ": ").strip()
    if not v and default is not None:
        return default
    return v


def _prompt_int(prompt: str, default: int) -> int:
    v = _prompt(prompt, str(default))
    try:
        return int(v)
    except Exception:
        return default


def _prompt_float(prompt: str, default: float) -> float:
    v = _prompt(prompt, str(default))
    try:
        return float(v)
    except Exception:
        return default


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        base_url = resolve_base_url(args.base_url)
        model = resolve_model(args.model)
        cfg = RunConfig(
            topic=args.topic,
            out=Path(args.out),
            base_url=base_url,
            api_key=resolve_api_key(),
            model=model,
            iterations=max(int(args.iterations), 1),
            max_sources=max(int(args.max_sources), 1),
            min_total_words=max(int(args.min_words), 1000),
            use_mock=bool(args.mock),
            verbose=bool(args.verbose or args.heartbeat),
            progress=bool(args.progress),
            llm_timeout_s=float(args.llm_timeout),
            request_budget_s=float(args.request_budget),
            keep_stage=bool(args.keep_stage),
            verbatim=bool(args.verbatim),
            archive_prompts=not bool(args.no_archive_prompts),
            archive_snapshots=bool(args.archive_snapshots),
            snapshot_timeout_s=float(args.snapshot_timeout),
            snapshot_total_timeout_s=float(args.snapshot_total_timeout),
            quality_gate=bool(args.quality_gate),
            min_quality_score=float(args.min_quality),
            max_quality_attempts=int(args.quality_attempts),
            seed_urls=args.seed_url,
        )
        run(cfg)
        return 0
    if args.cmd == "pre":
        from hydradeck.presets.rynnbrain import generate

        if str(args.preset).strip().lower() != "rynnbrain":
            print(f"Unknown preset: {args.preset}", file=sys.stderr)
            return 2
        generate(
            out=Path(args.out),
            keep_stage=bool(args.keep_stage),
            fetch=not bool(args.no_fetch),
        )
        return 0

    if args.cmd == "models":
        from hydradeck.clients import GrokClient

        client = GrokClient(
            base_url=resolve_base_url(str(args.base_url) if args.base_url else None),
            api_key=resolve_api_key(),
            model="grok-4",
        )
        for mid in client.list_models():
            print(mid)
        return 0

    if args.cmd == "auto":
        base_url = resolve_base_url(args.base_url)
        model = resolve_model(args.model)
        cfg = RunConfig(
            topic=args.topic,
            out=Path(args.out),
            base_url=base_url,
            api_key=resolve_api_key(),
            model=model,
            iterations=max(int(args.iterations), 1),
            max_sources=max(int(args.max_sources), 1),
            module_sources=max(int(args.module_sources), 1),
            query_count=max(int(args.query_count), 1),
            max_query_modules=max(int(args.max_query_modules), 1),
            sources_attempts=min(max(int(args.sources_attempts), 1), 3),
            facts_max_pages=max(int(args.facts_max_pages), 1),
            facts_max_chars_per_page=max(int(args.facts_max_chars), 1000),
            facts_target=max(int(args.facts_target), 5),
            judge_max_chars=max(int(args.judge_max_chars), 2000),
            max_total_runtime_s=float(args.max_runtime),
            min_total_words=12000,
            use_mock=bool(args.mock),
            verbose=bool(args.verbose or args.heartbeat),
            progress=bool(args.progress),
            llm_timeout_s=float(args.llm_timeout),
            keep_stage=False,
            verbatim=True,
            archive_prompts=True,
            archive_snapshots=True,
            snapshot_timeout_s=float(args.snapshot_timeout),
            auto=True,
            auto_queries=True,
            auto_models=True,
            quality_gate=True,
            min_quality_score=float(args.min_quality),
            max_quality_attempts=int(args.quality_attempts),
            seed_urls=None,
        )
        run(cfg)
        return 0

    if args.cmd == "config":
        uc = UserConfig(
            base_url=str(args.base_url) if args.base_url else None,
            api_key=str(args.api_key) if args.api_key else None,
            model=str(args.model) if args.model else None,
            pdf_compiler=str(args.pdf_compiler) if args.pdf_compiler else None,
            template=str(args.template) if args.template else None,
        )
        p = save_config(uc)
        print(str(p))
        return 0

    if args.cmd == "resources":
        base_url = resolve_base_url(args.base_url)
        model = resolve_model(args.model)
        cfg = RunConfig(
            topic=args.topic,
            out=Path(args.out),
            base_url=base_url,
            api_key=resolve_api_key(),
            model=model,
            pdf_compiler=str(args.pdf_compiler),
            template=str(args.template),
            max_sources=max(int(args.max_sources), 1),
            module_sources=max(int(args.module_sources), 1),
            use_mock=False,
            verbose=bool(args.heartbeat),
            progress=bool(args.progress),
            llm_timeout_s=float(args.llm_timeout),
            snapshot_timeout_s=float(args.snapshot_timeout),
            max_total_runtime_s=float(args.max_runtime),
            request_budget_s=float(args.request_budget),
            keep_stage=bool(args.keep_stage),
        )
        build_resources_pack(cfg)
        return 0

    if args.cmd == "wizard":
        topic = _prompt("Topic", "RynnBrain")
        out = args.out or _prompt("Output path (.zip)", "hydradeck/out/pre.zip")
        base_url = _prompt("Base URL (from config if empty)", "")
        model = _prompt("Model (from config if empty)", "")
        max_sources = _prompt_int("Max sources", 8)
        module_sources = _prompt_int("Sources per module", 3)
        llm_timeout = _prompt_float("LLM timeout (s)", 35.0)
        snapshot_timeout = _prompt_float("Snapshot timeout (s)", 10.0)
        max_runtime = _prompt_float("Max runtime (s)", 300.0)
        request_budget = _prompt_float("Per-request budget (s)", 20.0)
        pdf_compiler = _prompt("PDF compiler (auto|latexonline|texlive)", "auto")
        template = _prompt("Template (iclr2026|plain)", "iclr2026")

        cfg = RunConfig(
            topic=topic,
            out=Path(out),
            base_url=resolve_base_url(base_url or None),
            api_key=resolve_api_key(),
            model=resolve_model(model or None),
            pdf_compiler=pdf_compiler,
            template=template,
            max_sources=max(max_sources, 1),
            module_sources=max(module_sources, 1),
            use_mock=False,
            verbose=True,
            progress=True,
            llm_timeout_s=llm_timeout,
            snapshot_timeout_s=snapshot_timeout,
            max_total_runtime_s=max_runtime,
            request_budget_s=request_budget,
            keep_stage=False,
        )
        build_resources_pack(cfg)
        print(out)
        return 0

    print(f"Unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
