from __future__ import annotations

from pathlib import Path

from hydradeck.config import UserConfig, load_config, load_merged_config, save_config


def test_save_and_load_config(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    save_config(
        UserConfig(
            base_url="https://x",
            api_key="k",
            model="m",
            pdf_compiler="auto",
            template="iclr2026",
        ),
        path=p,
    )
    cfg = load_config(path=p)
    assert cfg.base_url == "https://x"
    assert cfg.api_key == "k"
    assert cfg.model == "m"
    assert cfg.pdf_compiler == "auto"
    assert cfg.template == "iclr2026"


def test_project_config_overrides_user(tmp_path: Path, monkeypatch) -> None:
    user_p = tmp_path / "user.json"
    save_config(UserConfig(base_url="https://u", api_key="u", model="u"), path=user_p)

    proj_root = tmp_path / "proj"
    (proj_root / ".hydradeck").mkdir(parents=True)
    proj_p = proj_root / ".hydradeck" / "config.json"
    save_config(UserConfig(model="p"), path=proj_p)

    monkeypatch.chdir(proj_root)
    from hydradeck import config as cfgmod

    monkeypatch.setattr(cfgmod, "config_path", lambda: user_p)
    merged = load_merged_config()
    assert merged.base_url == "https://u"
    assert merged.api_key == "u"
    assert merged.model == "p"
