---
title: HydraDeck
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# hydradeck

一个可重复、可审计的 Grok Deep Research 流水线（多 Persona 迭代），输出：

- `pre_report.md`：Pre-Research（研究前置）报告：研究问题拆解、方法、检索策略、风险与边界
- `report.md`：完整研究报告（含“完整资源”列表与可追溯引用）
- `speech.md`：演讲稿（可直接照读，含转场与时间提示）
- `pre_paper.tex`：Pre-brief 的 LaTeX 论文稿（article）
- `pre_slides.tex`：Pre-brief 的 Beamer 幻灯片
- `refs.bib`：BibTeX 参考文献
- `research.json`：结构化中间产物（便于复现与审计）

> 安全提示：不要把 API Key 写进仓库。请使用环境变量 `GROK_API_KEY`。
> 如果你已经在聊天里粘贴过 key，请立即**轮换/作废**该 key。

## 安装

```bash
cd hydradeck
python3 -m pip install -e .
python3 -m pip install -e ".[dev]"
```

## 快速使用

### 1) Mock（离线）跑通流程

```bash
mkdir -p out
hydradeck run --topic "LLM agents for deep research" --out out/demo.zip --mock
```

### 2) 使用 Grok2API / OpenAI 兼容网关

`api.example.com` 基于 Grok2API，提供 OpenAI 兼容的 `/v1/chat/completions` 与 `/v1/models`。

```bash
export GROK_BASE_URL="https://api.example.com"
export GROK_API_KEY="<YOUR_KEY>"
export GROK_MODEL="grok-4"

mkdir -p out
hydradeck run --topic "<你的研究主题>" --out out/topic.zip \
  --iterations 3 \
  --max-sources 10
```

## 输出结构

输出为一个目录或 zip（取决于 `--out` 是否以 `.zip` 结尾）。其中包含 `compile.sh` 与 `Makefile` 便于编译 LaTeX。

## WebUI（HydraDeck）

### 启动方式（本地）

```bash
cd hydradeck
python3 custom_web.py
```

默认监听：`http://127.0.0.1:7861`

> 说明：HydraDeck 仅保留 FastAPI WebUI，不再使用 Gradio。

### 运行前环境变量（可选）

```bash
export GROK_BASE_URL="https://api.example.com"
export GROK_API_KEY="<YOUR_KEY>"
export GROK_MODEL="grok-4"
```

### 页面基本使用

1. 在 `Run` 标签填写 Topic
2. 点 `Quick API Check` 先检查连通性
3. 点 `Run HydraDeck` 开始生成
4. 在 `Console` 查看实时进度
5. 在 `Artifacts` 下载 `paper.pdf` / `slides.pdf`
