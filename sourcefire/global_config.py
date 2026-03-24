"""Global Sourcefire configuration — stored in ~/.config/sourcefire/

This directory holds user-level settings (API keys, preferences) that
apply across all projects. Separate from the per-project .sourcefire/
directory which holds index data and project config.

On uninstall, `sourcefire --uninstall` removes this directory.
"""

from __future__ import annotations

import os
import platform
import tomllib
from pathlib import Path

import tomli_w


def get_global_dir() -> Path:
    """Return the global Sourcefire config directory.

    - macOS/Linux: ~/.config/sourcefire/
    - Windows: %APPDATA%/sourcefire/
    """
    if platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "sourcefire"
    return Path.home() / ".config" / "sourcefire"


def get_global_config_path() -> Path:
    return get_global_dir() / "config.toml"


def load_global_config() -> dict:
    """Load global config. Returns empty dict if not found."""
    path = get_global_config_path()
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_global_config(data: dict) -> None:
    """Save global config."""
    path = get_global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(data), encoding="utf-8")


def get_api_key() -> str:
    """Get the Gemini API key. Checks in order:

    1. GEMINI_API_KEY environment variable
    2. Global config (~/.sourcefire/config.toml)
    """
    # 1. Environment variable
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key

    # 2. Global config
    config = load_global_config()
    return config.get("gemini_api_key", "")


def save_api_key(key: str) -> None:
    """Save API key to global config."""
    config = load_global_config()
    config["gemini_api_key"] = key
    save_global_config(config)
    os.environ["GEMINI_API_KEY"] = key


def uninstall() -> None:
    """Remove the global ~/.sourcefire/ directory."""
    import shutil

    global_dir = get_global_dir()
    if global_dir.is_dir():
        shutil.rmtree(global_dir)
        print(f"Removed {global_dir}")
    else:
        print(f"Nothing to remove — {global_dir} does not exist.")
