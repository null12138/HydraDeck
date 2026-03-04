from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from hydradeck.packaging import finalize_output, stage_dir_for_out


@dataclass(frozen=True)
class PresetSource:
    url: str
    title: str
    kind: str
    priority: int
    notes: str


def _slugify(s: str) -> str:
    t = s.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = re.sub(r"-+", "-", t).strip("-")
    return t or "source"


def _fetch_snapshot(url: str, timeout_s: float = 25.0) -> tuple[str, str]:
    r = requests.get(url, timeout=timeout_s, headers={"User-Agent": "hydradeck/0.1"})
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    text = r.text
    if len(text) > 200_000:
        text = text[:200_000]
    return ctype, text


def _write_compile_helpers(out_dir: Path) -> None:
    _ = (out_dir / "compile.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "pdflatex -interaction=nonstopmode paper.tex",
                "bibtex paper || true",
                "pdflatex -interaction=nonstopmode paper.tex",
                "pdflatex -interaction=nonstopmode paper.tex",
                "pdflatex -interaction=nonstopmode slides.tex",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        (out_dir / "compile.sh").chmod(0o755)
    except Exception:
        pass
    _ = (out_dir / "Makefile").write_text(
        "".join(
            [
                "all: paper slides\n\n",
                "paper:\n\t",
                "pdflatex -interaction=nonstopmode paper.tex\n\t",
                "bibtex paper || true\n\t",
                "pdflatex -interaction=nonstopmode paper.tex\n\t",
                "pdflatex -interaction=nonstopmode paper.tex\n\n",
                "slides:\n\t",
                "pdflatex -interaction=nonstopmode slides.tex\n\n",
                "clean:\n\t",
                "rm -f *.aux *.bbl *.blg *.log *.out *.toc *.nav *.snm *.vrb *.fls *.fdb_latexmk\n",
            ]
        ),
        encoding="utf-8",
    )


def sources() -> list[PresetSource]:
    return [
        PresetSource(
            url="https://github.com/alibaba-damo-academy/RynnBrain",
            title="alibaba-damo-academy/RynnBrain (GitHub)",
            kind="primary",
            priority=1,
            notes="Code, checkpoints pointers, cookbooks, benchmarks.",
        ),
        PresetSource(
            url="https://alibaba-damo-academy.github.io/RynnBrain.github.io/",
            title="RynnBrain project page",
            kind="primary",
            priority=1,
            notes="Abstract, model lineup, demos, links.",
        ),
        PresetSource(
            url="https://arxiv.org/abs/2602.14979",
            title="RynnBrain: Open Embodied Foundation Models (arXiv:2602.14979)",
            kind="primary",
            priority=1,
            notes="Technical report; claims, methodology, evaluations.",
        ),
        PresetSource(
            url="https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-2B",
            title="RynnBrain-2B model card (Hugging Face)",
            kind="primary",
            priority=2,
            notes="Weights access, inference notes, license.",
        ),
        PresetSource(
            url="https://www.scmp.com/tech/tech-war/article/3343212/alibaba-unveils-rynnbrain-embodied-ai-model-gives-robots-brain",
            title="SCMP coverage: Alibaba unveils RynnBrain",
            kind="secondary",
            priority=3,
            notes="Press summary; may include comparisons and quotes.",
        ),
        PresetSource(
            url="https://connectcx.ai/alibabas-rynnbrain-advances-robot-intelligence/",
            title="CONNECTCX coverage: Alibaba’s RynnBrain Advances Robot Intelligence",
            kind="secondary",
            priority=4,
            notes="Third-party coverage; validate against primary sources.",
        ),
        PresetSource(
            url="https://huggingface.co/papers/2602.14979",
            title="Hugging Face Papers page for arXiv:2602.14979",
            kind="secondary",
            priority=4,
            notes="Convenient summary + links.",
        ),
    ]


def pre_report_md() -> str:
    srcs = sources()
    src_lines = [
        "\n".join(
            [
                f"[{i}] {s.title}",
                f"    - URL: {s.url}",
                f"    - Type: {s.kind} | Priority: {s.priority}",
                f"    - Notes: {s.notes}",
            ]
        )
        for i, s in enumerate(srcs, start=1)
    ]
    queries = [
        "RynnBrain arXiv 2602.14979 benchmark 16 leaderboards details",
        "RynnBrain 30B-A3B MoE architecture A3B meaning experts routing",
        "RynnBrain spatiotemporal grounding egocentric cognition definitions",
        "RynnBrain-Plan manipulation planning dataset tasks evaluation",
        "RynnBrain-Nav VLN benchmarks used and results",
        "RynnBrain-CoP chain-of-point spatial reasoning prompt format",
        "Qwen3-VL base model differences vs RynnBrain modifications",
        "Embodied foundation model comparison: Gemini Robotics ER 1.5 Cosmos Reason 2",
        "Licensing: Apache-2.0 weights usage restrictions if any",
        "Reproducibility: official code inference requirements and compute",
    ]

    talk = [
        "0:00–1:30 目标与背景：什么是 embodied foundation model，RynnBrain 想解决什么问题",
        "1:30–4:30 一手资料快速过一遍：GitHub / Project Page / arXiv（只提我们要验证的关键点）",
        "4:30–7:30 研究问题拆解：能力维度（感知/记忆/定位/推理/规划）",
        "           与任务维度（nav/manipulation）",
        "7:30–10:30 证据计划：哪些 claim 必须用什么证据验证",
        "            （leaderboard、消融、数据集、代码可复现性）",
        "10:30–13:00 风险与不确定性：宣传与论文差异、评测口径、demo bias、实现门槛",
        "13:00–15:00 输出计划：最终报告结构、资源打包、可复现 checklist",
    ]

    return "\n".join(
        [
            "# Pre-Research (15min) — RynnBrain",
            "",
            "本 Pre-Research 的目标不是给出最终结论，而是建立**可验证的研究路线**：",
            "明确问题、证据标准、资源与时间安排，确保后续 deep research 不会变成‘看 demo 写总结’。",
            "",
            "## 1. 15 分钟口头 Pre-Brief 讲稿大纲（可照读）",
            "\n".join([f"- {x}" for x in talk]),
            "",
            "## 2. 研究对象界定（Working definition）",
            "- RynnBrain 是 Alibaba DAMO Academy 在 2026 年 2 月左右开源的一套",
            "  embodied foundation model 家族。",
            "- 它强调：以第一人称/自我中心（egocentric）视角做理解，具备时空定位/记忆",
            "  （spatiotemporal grounding / memory），并面向真实任务规划（planning）。",
            "- 需要通过一手材料确认：模型族谱（2B/8B/30B MoE，以及 Plan/Nav/CoP 等子模型）、",
            "  评测体系、训练数据与推理方式，以及开源范围（代码/权重/benchmark）。",
            "",
            "## 3. 研究问题（Research Questions）",
            "下面的问题按优先级排序，前 3 个属于‘不解决就不要写结论’：",
            "",
            "### RQ1（最高优先级）：RynnBrain 的核心技术增量是什么？",
            "- 相比 Qwen3-VL 等基础 VLM，它到底加了什么：时空记忆模块？定位/地图表征？",
            "  多任务 head？还是主要靠数据与训练配方？",
            "- 需要在 arXiv 技术报告里找到：架构图、训练目标、数据组成、消融实验。",
            "",
            "### RQ2：‘SOTA on 16 embodied leaderboards’ 这类 claim 的证据链是否站得住？",
            "- 需要明确：16 个榜单各自是什么任务/指标/基线；是否同一评测口径；",
            "  是否存在 cherry-pick。",
            "- 证据标准：必须来自官方 benchmark 页面/leaderboard 截图/可复现脚本，而不是新闻稿。",
            "",
            "### RQ3：开源的可用性如何（工程落地门槛）？",
            "- 权重是否全量公开？推理依赖（框架版本、显存、是否需要视频输入管线）？",
            "- 是否提供 cookbooks，覆盖哪些能力：定位、推理、规划、导航、操作。",
            "",
            "### RQ4：能力维度拆解：它到底在‘什么能力’上强？",
            "- Egocentric cognition：是否包含长期场景理解与一致性跟踪？",
            "- Spatiotemporal grounding：是否输出坐标/轨迹/地图？误差量化如何做？",
            "- Planning：是语言层规划（plan-as-text），还是能输出可执行动作序列",
            "  （actions/waypoints）？",
            "",
            "### RQ5：与同类系统的可比性（apples-to-apples）",
            "- 对比对象：Gemini Robotics ER、NVIDIA Cosmos Reason、其它 embodied VLM / EFM。",
            "- 对比口径：任务集/传感器输入/是否允许工具调用/是否闭源系统。",
            "",
            "## 4. Scope / Non-Scope（边界）",
            "### Scope",
            "- 以公开资料为边界：论文/项目页/代码/模型卡/公开 benchmark。",
            "- 产出一个可审计的‘证据 → 结论’矩阵：每个结论都对应来源与验证步骤。",
            "",
            "### Non-Scope（本轮明确不做）",
            "- 不做真实机器人部署复现（除非官方提供可运行 demo 且成本可控）。",
            "- 不做未公开数据/内部实现猜测；不引用无法访问或不可验证的泄漏信息。",
            "",
            "## 5. 证据标准（Evaluation Criteria）",
            "为了避免‘看起来很强’的主观总结，本研究采用硬标准：",
            "- 论文证据：架构/训练/消融/实验设置必须可在 arXiv 报告中定位到章节与图表。",
            "- 代码证据：能在 GitHub 找到对应实现入口（推理脚本、配置、模型定义）。",
            "- Bench 证据：结果必须能追溯到官方 benchmark/leaderboard 或可复现评测脚本。",
            "- 口径一致：比较必须满足相同输入与评测规则；否则标注为‘不可直接比较’。",
            "- 可用性：给出最小可运行路径（依赖、命令、显存、样例输入）。",
            "",
            "## 6. 检索与阅读计划（Search Plan & Reading Plan）",
            "### 6.1 顺序（建议在 2–4 小时深研里执行）",
            "1) GitHub README + 目录：确定开源范围、模型列表、入口脚本、benchmark 链接。\n"
            "2) Project Page：收集所有外链（HF/ModelScope/Benchmark/Demo/Video）。\n"
            "3) arXiv：抓核心章节：method、experiments、ablation、limitations。\n"
            "4) Model Card：确认权重、许可证、推理限制与样例。\n"
            "5) Press：只作为线索，不作为证据；对 press 中的 claim 做反向核对。",
            "",
            "### 6.2 Query 列表（可直接用于搜索/对照阅读）",
            "\n".join([f"- {q}" for q in queries]),
            "",
            "## 7. 产出设计（Deliverables）",
            "在完成 deep research 后，最终交付物建议包含：",
            "- 长文研究报告（含 Resources、证据矩阵、可复现路径、局限与开放问题）",
            "- 15 分钟演讲稿 + Beamer（信息密度高，但每页只承载一个结论）",
            "- research.json（结构化审计：来源、摘录、结论、证据链接、验证状态）",
            "- resources/（把关键页面快照打包，避免链接失效）",
            "",
            "## 8. 风险与不确定性（Risks & Unknowns）",
            "- Press 可能夸大：需以论文与 benchmark 为准。",
            "- Leaderboard 的口径可能不统一：需逐项核对设置。",
            "- Demo bias：演示视频不等于泛化能力。",
            "- 可复现门槛：依赖、算力、输入管线（视频/多帧）可能较重。",
            "- 许可证与权重条款：代码 Apache-2.0 不等于所有权重都无约束。",
            "",
            "## 9. 资源清单（Prioritized Resources）",
            "\n".join(src_lines),
            "",
        ]
    )


def generate(out: Path, keep_stage: bool, fetch: bool) -> Path:
    stage_dir = stage_dir_for_out(out)
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_compile_helpers(stage_dir)

    srcs = sources()
    src_json = [asdict(s) for s in srcs]

    resources_dir = stage_dir / "resources"
    snapshots_dir = resources_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    _ = (resources_dir / "sources.json").write_text(
        json.dumps({"sources": src_json}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    snapshots: list[dict[str, object]] = []
    if fetch:
        for i, s in enumerate(srcs, start=1):
            slug = _slugify(s.title)
            target = snapshots_dir / f"{i:02d}_{slug}.txt"
            entry: dict[str, object] = {"url": s.url, "title": s.title, "path": str(target)}
            try:
                ctype, text = _fetch_snapshot(s.url)
                entry["content_type"] = ctype
                _ = target.write_text(text, encoding="utf-8")
                entry["ok"] = True
            except Exception as e:
                entry["ok"] = False
                entry["error"] = str(e)
            snapshots.append(entry)

    pre = pre_report_md()
    _ = (stage_dir / "pre_report.md").write_text(pre, encoding="utf-8")
    _ = (stage_dir / "report.md").write_text("# (Not generated in preset mode)\n", encoding="utf-8")
    _ = (stage_dir / "speech.md").write_text("# (Not generated in preset mode)\n", encoding="utf-8")
    _ = (stage_dir / "paper.tex").write_text(
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[UTF8]{ctex}\n"
        "\\usepackage{hyperref}\n"
        "\\title{RynnBrain Pre-Research}\n"
        "\\author{hydradeck preset}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\section*{Pre-Research}\n"
        "This preset package contains a Markdown pre-research report and archived resources.\\\\\n"
        "See pre_report.md and resources/.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    _ = (stage_dir / "slides.tex").write_text(
        "\\documentclass{beamer}\n"
        "\\usepackage[UTF8]{ctex}\n"
        "\\usetheme{Madrid}\n"
        "\\title{RynnBrain Pre-Research (15min)}\n"
        "\\author{hydradeck preset}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\frame{\\titlepage}\n"
        "\\begin{frame}{What is inside?}\n"
        "- pre_report.md\\\\\n"
        "- resources/sources.json\\\\\n"
        "- resources/snapshots/*\\\\\n"
        "\\end{frame}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    _ = (stage_dir / "refs.bib").write_text("% (Not generated in preset mode)\n", encoding="utf-8")

    research = {
        "topic": "RynnBrain",
        "mode": "preset-pre",
        "sources": src_json,
        "snapshots": snapshots,
        "meta": {"fetch": fetch},
    }
    _ = (stage_dir / "research.json").write_text(
        json.dumps(research, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    finalize_output(out, stage_dir, keep_stage=keep_stage)
    return out
