from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    topic: str
    out: Path
    base_url: str
    api_key: str
    model: str

    iterations: int = 3
    max_sources: int = 10
    module_sources: int = 4
    min_total_words: int = 12000

    use_mock: bool = False
    verbose: bool = False

    llm_timeout_s: float = 180.0
    facts_max_pages: int = 6
    facts_max_chars_per_page: int = 8000
    facts_target: int = 40

    judge_max_chars: int = 12000

    pre_tex_quality_gate: bool = True
    pre_tex_min_score: float = 0.85
    pre_tex_attempts: int = 2
    keep_stage: bool = False
    verbatim: bool = False
    archive_prompts: bool = True

    archive_snapshots: bool = False
    snapshot_timeout_s: float = 25.0
    snapshot_total_timeout_s: float = 60.0

    auto: bool = False
    auto_queries: bool = False
    auto_models: bool = False

    quality_gate: bool = False
    min_quality_score: float = 0.85
    max_quality_attempts: int = 3

    query_count: int = 10
    max_query_modules: int = 3

    sources_attempts: int = 3

    max_total_runtime_s: float = 240.0

    progress: bool = False

    request_budget_s: float = 20.0

    pdf_compiler: str = "auto"

    template: str = "pretty"

    seed_urls: list[str] | None = None


@dataclass(frozen=True)
class Source:
    url: str
    title: str
    snippet: str


@dataclass(frozen=True)
class ExtractedFact:
    claim: str
    evidence: str
    url: str
    title: str


@dataclass(frozen=True)
class ResearchOutputs:
    pre_report_md: str
    report_md: str
    speech_md: str
    paper_tex: str
    slides_tex: str
    bibtex: str
    meta: dict[str, Any]
