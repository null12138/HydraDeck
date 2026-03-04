from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UserConfig:
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    pdf_compiler: str | None = None
    template: str | None = None


def config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "hydradeck" / "config.json"
    return Path.home() / ".config" / "hydradeck" / "config.json"


def load_config(path: Path | None = None) -> UserConfig:
    p = path or config_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return UserConfig()
    if not isinstance(data, dict):
        return UserConfig()
    base_url = data.get("base_url")
    api_key = data.get("api_key")
    model = data.get("model")
    pdf_compiler = data.get("pdf_compiler")
    template = data.get("template")
    return UserConfig(
        base_url=base_url if isinstance(base_url, str) else None,
        api_key=api_key if isinstance(api_key, str) else None,
        model=model if isinstance(model, str) else None,
        pdf_compiler=pdf_compiler if isinstance(pdf_compiler, str) else None,
        template=template if isinstance(template, str) else None,
    )


def find_project_config(start: Path | None = None) -> Path | None:
    cur = (start or Path.cwd()).resolve()
    for _ in range(8):
        cand = cur / ".hydradeck" / "config.json"
        if cand.exists():
            return cand
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def load_merged_config() -> UserConfig:
    user = load_config()
    pc = find_project_config()
    if pc is None:
        return user
    proj = load_config(path=pc)
    return UserConfig(
        base_url=proj.base_url or user.base_url,
        api_key=proj.api_key or user.api_key,
        model=proj.model or user.model,
        pdf_compiler=proj.pdf_compiler or user.pdf_compiler,
        template=proj.template or user.template,
    )


def save_config(cfg: UserConfig, path: Path | None = None) -> Path:
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if cfg.base_url:
        payload["base_url"] = cfg.base_url
    if cfg.api_key:
        payload["api_key"] = cfg.api_key
    if cfg.model:
        payload["model"] = cfg.model
    if cfg.pdf_compiler:
        payload["pdf_compiler"] = cfg.pdf_compiler
    if cfg.template:
        payload["template"] = cfg.template
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def resolve_api_key() -> str:
    env = os.environ.get("GROK_API_KEY")
    if env:
        return env
    cfg = load_merged_config()
    return cfg.api_key or ""


def resolve_base_url(default: str | None = None) -> str:
    env = os.environ.get("GROK_BASE_URL")
    if env:
        return env
    cfg = load_merged_config()
    if cfg.base_url:
        return cfg.base_url
    if default is None:
        raise RuntimeError("Missing base_url: set GROK_BASE_URL or hydradeck config --base-url")
    return default


def resolve_model(default: str | None = None) -> str:
    env = os.environ.get("GROK_MODEL")
    if env:
        return env
    cfg = load_merged_config()
    if cfg.model:
        return cfg.model
    if default is None:
        raise RuntimeError("Missing model: set GROK_MODEL or hydradeck config --model")
    return default


def resolve_pdf_compiler(default: str) -> str:
    env = os.environ.get("HYDRADECK_PDF_COMPILER")
    if env:
        return env
    cfg = load_merged_config()
    return cfg.pdf_compiler or default


def resolve_template(default: str) -> str:
    env = os.environ.get("HYDRADECK_TEMPLATE")
    if env:
        return env
    cfg = load_merged_config()
    return cfg.template or default
