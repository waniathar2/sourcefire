"""Auto-initialization for Sourcefire — creates .sourcefire/ with LLM-generated config."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

from sourcefire.config import SourcefireConfig, default_config, save_config
from sourcefire.indexer.language_profiles import get_profile


def scan_file_tree(project_dir: Path, max_files: int = 5000) -> str:
    """Scan the project directory and return a text representation of the file tree."""
    skip_dirs = {
        ".git", "node_modules", "__pycache__", "build", "dist", "target",
        ".dart_tool", ".next", "venv", ".venv", ".idea", ".vs", ".sourcefire",
        ".tox", "eggs",
    }

    lines: list[str] = []
    file_count = 0

    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]

        rel_root = Path(root).relative_to(project_dir).as_posix()
        if rel_root == ".":
            rel_root = ""

        for f in sorted(files):
            if file_count >= max_files:
                break
            rel_path = f"{rel_root}/{f}" if rel_root else f
            lines.append(rel_path)
            file_count += 1

        if file_count >= max_files:
            break

    return "\n".join(lines)


def _generate_patterns_via_llm(file_tree: str, api_key: str) -> dict[str, list[str]] | None:
    """Ask the LLM to generate include/exclude patterns from the file tree."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
        )

        prompt = (
            "Given this project file tree, determine which files are source code worth "
            "indexing for a code RAG system. Respond with ONLY a TOML code block containing "
            "two arrays: `include` (glob patterns for source files, configs, and docs) and "
            "`exclude` (glob patterns for build artifacts, dependencies, generated files, "
            "and non-code assets). Be comprehensive but conservative.\n\n"
            "Always include these in exclude: .git/**, .sourcefire/**\n\n"
            f"```\n{file_tree}\n```"
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)

        match = re.search(r"```(?:toml)?\s*\n(.*?)\n```", text, re.DOTALL)
        if not match:
            return None

        toml_str = match.group(1)
        data = tomllib.loads(toml_str)

        include = data.get("include", [])
        exclude = data.get("exclude", [])

        if isinstance(include, list) and isinstance(exclude, list):
            if ".sourcefire/**" not in exclude:
                exclude.append(".sourcefire/**")
            return {"include": include, "exclude": exclude}

        return None

    except Exception as exc:
        print(f"[init] LLM config generation failed: {exc}")
        return None


def _fallback_patterns(project_dir: Path, language_override: str | None = None) -> dict[str, list[str]]:
    """Generate patterns from language profile when LLM is unavailable."""
    profile = get_profile(project_dir, language_override)

    if profile:
        include = list(profile.include_patterns)
        exclude = list(profile.exclude_patterns)
    else:
        include = ["**/*"]
        exclude = []

    for pat in [".git/**", ".sourcefire/**", "node_modules/**", "__pycache__/**",
                "*.pyc", ".venv/**", "venv/**", "dist/**", "build/**"]:
        if pat not in exclude:
            exclude.append(pat)

    for pat in ["README.md", "CLAUDE.md"]:
        if pat not in include:
            include.append(pat)

    return {"include": include, "exclude": exclude}


def auto_init(
    project_dir: Path,
    sourcefire_dir: Path | None = None,
    api_key: str = "",
    language_override: str | None = None,
) -> SourcefireConfig:
    """Initialize .sourcefire/ directory with LLM-generated config."""
    if sourcefire_dir is None:
        sourcefire_dir = project_dir / ".sourcefire"

    print(f"[init] Initializing Sourcefire for: {project_dir.name}")

    sourcefire_dir.mkdir(parents=True, exist_ok=True)

    print("[init] Scanning project structure...")
    file_tree = scan_file_tree(project_dir)

    patterns: dict[str, list[str]] | None = None
    if api_key:
        print("[init] Generating config via LLM...")
        patterns = _generate_patterns_via_llm(file_tree, api_key)

    if patterns:
        print(f"[init] LLM generated {len(patterns['include'])} include, {len(patterns['exclude'])} exclude patterns.")
    else:
        print("[init] Using language-profile defaults for patterns.")
        patterns = _fallback_patterns(project_dir, language_override)

    profile = get_profile(project_dir, language_override)
    language = profile.language if profile else "auto"

    config = default_config(project_dir)
    config.sourcefire_dir = sourcefire_dir
    config.include = patterns["include"]
    config.exclude = patterns["exclude"]
    config.language = language

    save_config(config)
    print(f"[init] Config written to: {config.config_path}")

    print("\nTip: Add to your .gitignore:")
    print("  .sourcefire/chroma/")
    print("  .sourcefire/graph.json")
    print("  .sourcefire/.lock\n")

    return config


def reinit_patterns(
    config: SourcefireConfig,
    api_key: str = "",
) -> SourcefireConfig:
    """Regenerate only the [indexer] include/exclude patterns, preserving other config."""
    print(f"[init] Regenerating patterns for: {config.project_dir.name}")

    file_tree = scan_file_tree(config.project_dir)

    patterns: dict[str, list[str]] | None = None
    if api_key:
        print("[init] Generating patterns via LLM...")
        patterns = _generate_patterns_via_llm(file_tree, api_key)

    if patterns:
        print(f"[init] LLM generated {len(patterns['include'])} include, {len(patterns['exclude'])} exclude patterns.")
    else:
        print("[init] Using language-profile defaults.")
        language_override = config.language if config.language != "auto" else None
        patterns = _fallback_patterns(config.project_dir, language_override)

    config.include = patterns["include"]
    config.exclude = patterns["exclude"]

    save_config(config)
    print(f"[init] Updated patterns in: {config.config_path}")

    return config
