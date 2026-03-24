"""Language profiles for language-agnostic code analysis.

Each profile defines the patterns, AST node types, and regex needed to
parse imports, exports, and chunk boundaries for a specific language.
Auto-detection scans the project directory to count file extensions and
picks the dominant language.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class LanguageProfile:
    """Configuration for language-specific code analysis."""

    language: str
    file_extensions: list[str]

    # File collection patterns
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)

    # Tree-sitter config (optional)
    tree_sitter_language: Optional[str] = None
    import_node_types: list[str] = field(default_factory=list)
    export_node_types: list[str] = field(default_factory=list)
    boundary_node_types: list[str] = field(default_factory=list)
    string_literal_type: str = "string_literal"

    # Regex fallback patterns
    import_pattern: Optional[str] = None
    export_pattern: Optional[str] = None
    boundary_pattern: Optional[str] = None

    # Path-based metadata inference (optional)
    layer_parts: list[str] = field(default_factory=list)
    feature_regex: Optional[str] = None
    file_type_suffixes: list[tuple[str, str]] = field(default_factory=list)
    directory_type_patterns: dict[str, str] = field(default_factory=dict)

    # Import graph config
    external_import_prefixes: tuple[str, ...] = ()

    # Syntax highlighting language for code blocks
    highlight_language: str = "text"

    # Stack trace / file reference regex patterns
    file_ref_patterns: list[str] = field(default_factory=list)

    # Search extensions for keyword search tool
    searchable_extensions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in language profiles
# ---------------------------------------------------------------------------

DART_PROFILE = LanguageProfile(
    language="dart",
    file_extensions=[".dart"],
    include_patterns=["lib/**/*.dart", "pubspec.yaml", "analysis_options.yaml"],
    exclude_patterns=["*.g.dart", "*.freezed.dart", "test/**"],
    tree_sitter_language="dart",
    import_node_types=["import_specification"],
    export_node_types=[
        "class_definition",
        "enum_declaration",
        "mixin_declaration",
        "extension_declaration",
    ],
    boundary_node_types=[
        "class_definition",
        "enum_declaration",
        "mixin_declaration",
        "extension_declaration",
    ],
    import_pattern=r"""import\s+'([^']+)'""",
    export_pattern=r"""^(?:abstract\s+)?(?:class|enum|mixin|extension)\s+(\w+)""",
    boundary_pattern=r"""^(?:abstract\s+)?(?:class|enum|mixin|extension)\s+\w+""",
    layer_parts=["presentation", "domain", "data", "core"],
    feature_regex=r"features/(\w+)/",
    file_type_suffixes=[
        ("_remote_datasource", "datasource"),
        ("_datasource", "datasource"),
        ("_repository_impl", "repository"),
        ("_repository", "repository"),
        ("_notifier", "notifier"),
        ("_provider", "provider"),
        ("_interceptor", "interceptor"),
        ("_screen", "screen"),
        ("_widget", "widget"),
        ("_model", "model"),
        ("_entity", "entity"),
    ],
    directory_type_patterns={"/widgets/": "widget"},
    external_import_prefixes=("package:", "dart:"),
    highlight_language="dart",
    file_ref_patterns=[
        r"package:[^/]+/((?:features|core|lib)[^\s:)]+\.dart):(\d+)",
        r"\b(lib/[^\s:)]+\.dart)(?::(\d+))?",
    ],
    searchable_extensions=[".dart", ".yaml", ".json", ".md"],
)

PYTHON_PROFILE = LanguageProfile(
    language="python",
    file_extensions=[".py"],
    include_patterns=["src/**/*.py", "**/*.py"],
    exclude_patterns=["__pycache__/**", "*.pyc", ".venv/**", "venv/**", "test/**", "tests/**"],
    tree_sitter_language="python",
    import_node_types=["import_statement", "import_from_statement"],
    export_node_types=["class_definition", "function_definition"],
    boundary_node_types=["class_definition", "function_definition"],
    import_pattern=r"""^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))""",
    export_pattern=r"""^(?:class|def|async\s+def)\s+(\w+)""",
    boundary_pattern=r"""^(?:class|def|async\s+def)\s+\w+""",
    layer_parts=["api", "services", "models", "core", "utils"],
    feature_regex=r"(?:features|modules|apps)/(\w+)/",
    file_type_suffixes=[
        ("_test", "test"),
        ("_service", "service"),
        ("_handler", "handler"),
        ("_model", "model"),
        ("_schema", "schema"),
        ("_router", "router"),
        ("_view", "view"),
        ("_serializer", "serializer"),
    ],
    external_import_prefixes=(),
    highlight_language="python",
    file_ref_patterns=[
        r'File "([^"]+\.py)", line (\d+)',
        r"\b([\w/]+\.py)(?::(\d+))?",
    ],
    searchable_extensions=[".py", ".yaml", ".yml", ".json", ".md", ".toml"],
)

JAVASCRIPT_PROFILE = LanguageProfile(
    language="javascript",
    file_extensions=[".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"],
    include_patterns=["src/**/*.{js,jsx,ts,tsx}", "**/*.{js,jsx,ts,tsx}"],
    exclude_patterns=["node_modules/**", "dist/**", "build/**", ".next/**", "*.test.*", "*.spec.*"],
    tree_sitter_language="typescript",
    import_node_types=["import_statement"],
    export_node_types=["export_statement", "class_declaration", "function_declaration"],
    boundary_node_types=["export_statement", "class_declaration", "function_declaration"],
    import_pattern=r"""(?:import|require)\s*\(?['"]([\w@./][^'"]*)['"]\)?""",
    export_pattern=r"""^(?:export\s+)?(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)""",
    boundary_pattern=r"""^(?:export\s+)?(?:default\s+)?(?:class|function|const|let|var)\s+\w+""",
    layer_parts=["components", "pages", "hooks", "services", "utils", "api", "lib"],
    feature_regex=r"(?:features|modules)/(\w+)/",
    file_type_suffixes=[
        (".test", "test"),
        (".spec", "test"),
        (".hook", "hook"),
        (".service", "service"),
        (".controller", "controller"),
        (".middleware", "middleware"),
        (".component", "component"),
        (".page", "page"),
        (".route", "route"),
    ],
    external_import_prefixes=(),
    highlight_language="typescript",
    file_ref_patterns=[
        r"\b(src/[^\s:)]+\.(?:ts|tsx|js|jsx))(?::(\d+))?",
    ],
    searchable_extensions=[".js", ".jsx", ".ts", ".tsx", ".json", ".md"],
)

GO_PROFILE = LanguageProfile(
    language="go",
    file_extensions=[".go"],
    include_patterns=["**/*.go"],
    exclude_patterns=["vendor/**", "*_test.go"],
    tree_sitter_language="go",
    import_node_types=["import_declaration"],
    export_node_types=["function_declaration", "type_declaration", "method_declaration"],
    boundary_node_types=["function_declaration", "type_declaration", "method_declaration"],
    import_pattern=r"""^\s*"([^"]+)"$""",
    export_pattern=r"""^(?:func|type)\s+(?:\(.*?\)\s+)?(\w+)""",
    boundary_pattern=r"""^(?:func|type)\s+""",
    layer_parts=["cmd", "internal", "pkg", "api"],
    feature_regex=r"(?:internal|pkg)/(\w+)/",
    file_type_suffixes=[
        ("_handler", "handler"),
        ("_service", "service"),
        ("_repository", "repository"),
        ("_model", "model"),
        ("_middleware", "middleware"),
    ],
    external_import_prefixes=(),
    highlight_language="go",
    file_ref_patterns=[
        r"\b([\w/]+\.go):(\d+)",
    ],
    searchable_extensions=[".go", ".yaml", ".yml", ".json", ".md"],
)

RUST_PROFILE = LanguageProfile(
    language="rust",
    file_extensions=[".rs"],
    include_patterns=["src/**/*.rs", "**/*.rs"],
    exclude_patterns=["target/**"],
    tree_sitter_language="rust",
    import_node_types=["use_declaration"],
    export_node_types=["function_item", "struct_item", "enum_item", "impl_item", "trait_item"],
    boundary_node_types=["function_item", "struct_item", "enum_item", "impl_item", "trait_item"],
    import_pattern=r"""^use\s+([\w:]+)""",
    export_pattern=r"""^(?:pub\s+)?(?:fn|struct|enum|impl|trait)\s+(\w+)""",
    boundary_pattern=r"""^(?:pub\s+)?(?:fn|struct|enum|impl|trait)\s+\w+""",
    layer_parts=["api", "domain", "infrastructure", "lib"],
    feature_regex=None,
    file_type_suffixes=[
        ("_test", "test"),
        ("_handler", "handler"),
        ("_service", "service"),
    ],
    external_import_prefixes=(),
    highlight_language="rust",
    file_ref_patterns=[
        r"\b([\w/]+\.rs):(\d+)",
    ],
    searchable_extensions=[".rs", ".toml", ".yaml", ".md"],
)

JAVA_PROFILE = LanguageProfile(
    language="java",
    file_extensions=[".java"],
    include_patterns=["src/**/*.java", "**/*.java"],
    exclude_patterns=["build/**", "target/**", "*Test.java", "*Tests.java"],
    tree_sitter_language="java",
    import_node_types=["import_declaration"],
    export_node_types=["class_declaration", "interface_declaration", "enum_declaration"],
    boundary_node_types=["class_declaration", "interface_declaration", "enum_declaration"],
    import_pattern=r"""^import\s+([\w.]+);""",
    export_pattern=r"""^(?:public\s+)?(?:class|interface|enum)\s+(\w+)""",
    boundary_pattern=r"""^(?:public\s+)?(?:class|interface|enum)\s+\w+""",
    layer_parts=["controller", "service", "repository", "model", "dto", "config"],
    feature_regex=None,
    file_type_suffixes=[
        ("Controller", "controller"),
        ("Service", "service"),
        ("Repository", "repository"),
        ("Dto", "dto"),
        ("Entity", "entity"),
        ("Config", "config"),
    ],
    external_import_prefixes=("java.", "javax.", "jakarta."),
    highlight_language="java",
    file_ref_patterns=[
        r"\b([\w/]+\.java):(\d+)",
    ],
    searchable_extensions=[".java", ".xml", ".yaml", ".yml", ".json", ".md"],
)

C_PROFILE = LanguageProfile(
    language="c",
    file_extensions=[".c", ".h"],
    include_patterns=["src/**/*.c", "src/**/*.h", "include/**/*.h", "**/*.c", "**/*.h"],
    exclude_patterns=["build/**", "cmake-build-*/**", "third_party/**", "vendor/**"],
    tree_sitter_language="c",
    import_node_types=["preproc_include"],
    export_node_types=["function_definition", "struct_specifier", "enum_specifier", "type_definition"],
    boundary_node_types=["function_definition", "struct_specifier", "enum_specifier", "type_definition"],
    import_pattern=r"""^#include\s+[<"]([\w/.]+)[>"]""",
    export_pattern=r"""^(?:static\s+)?(?:inline\s+)?(?:extern\s+)?(?:\w+[\s*]+)+(\w+)\s*\(""",
    boundary_pattern=r"""^(?:static\s+)?(?:inline\s+)?(?:extern\s+)?(?:\w+[\s*]+)+\w+\s*\(|^(?:typedef\s+)?(?:struct|enum|union)\s+\w+""",
    layer_parts=["src", "include", "lib", "drivers", "core", "hal"],
    feature_regex=r"(?:src|modules)/(\w+)/",
    file_type_suffixes=[
        ("_test", "test"),
        ("_hal", "hal"),
        ("_driver", "driver"),
        ("_util", "util"),
    ],
    external_import_prefixes=(),
    highlight_language="c",
    file_ref_patterns=[
        r"\b([\w/]+\.[ch]):(\d+)",
    ],
    searchable_extensions=[".c", ".h", ".md", ".txt", ".cmake"],
)

CPP_PROFILE = LanguageProfile(
    language="cpp",
    file_extensions=[".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".hh", ".h"],
    include_patterns=[
        "src/**/*.cpp", "src/**/*.cc", "src/**/*.cxx",
        "src/**/*.hpp", "src/**/*.hxx", "src/**/*.hh",
        "include/**/*.hpp", "include/**/*.hxx", "include/**/*.hh", "include/**/*.h",
        "**/*.cpp", "**/*.cc", "**/*.hpp",
    ],
    exclude_patterns=["build/**", "cmake-build-*/**", "third_party/**", "vendor/**"],
    tree_sitter_language="cpp",
    import_node_types=["preproc_include"],
    export_node_types=[
        "function_definition", "class_specifier", "struct_specifier",
        "enum_specifier", "namespace_definition", "template_declaration",
    ],
    boundary_node_types=[
        "function_definition", "class_specifier", "struct_specifier",
        "enum_specifier", "namespace_definition", "template_declaration",
    ],
    import_pattern=r"""^#include\s+[<"]([\w/.]+)[>"]""",
    export_pattern=r"""^(?:template\s*<[^>]*>\s*)?(?:class|struct|enum(?:\s+class)?|namespace)\s+(\w+)|^(?:[\w:*&<>\s]+)\s+(\w+)\s*\(""",
    boundary_pattern=r"""^(?:template\s*<[^>]*>\s*)?(?:class|struct|enum|namespace)\s+\w+|^(?:[\w:*&<>\s]+)\s+\w+\s*\(""",
    layer_parts=["src", "include", "lib", "core", "engine", "modules"],
    feature_regex=r"(?:src|modules)/(\w+)/",
    file_type_suffixes=[
        ("_test", "test"),
        ("_impl", "implementation"),
        ("_factory", "factory"),
        ("_manager", "manager"),
        ("_handler", "handler"),
        ("_util", "util"),
    ],
    external_import_prefixes=(),
    highlight_language="cpp",
    file_ref_patterns=[
        r"\b([\w/]+\.(?:cpp|cc|cxx|hpp|hxx|hh|h)):(\d+)",
    ],
    searchable_extensions=[".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".hh", ".h", ".cmake", ".md"],
)

# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "dart": DART_PROFILE,
    "python": PYTHON_PROFILE,
    "javascript": JAVASCRIPT_PROFILE,
    "typescript": JAVASCRIPT_PROFILE,
    "go": GO_PROFILE,
    "rust": RUST_PROFILE,
    "java": JAVA_PROFILE,
    "c": C_PROFILE,
    "cpp": CPP_PROFILE,
}

# Map file extension -> language name for quick lookup
_EXTENSION_TO_LANGUAGE: dict[str, str] = {}
for _name, _profile in LANGUAGE_PROFILES.items():
    for _ext in _profile.file_extensions:
        _EXTENSION_TO_LANGUAGE.setdefault(_ext, _name)


def get_profile_for_extension(ext: str) -> Optional[LanguageProfile]:
    """Return the language profile for a given file extension, or None."""
    lang = _EXTENSION_TO_LANGUAGE.get(ext)
    return LANGUAGE_PROFILES.get(lang) if lang else None


# ---------------------------------------------------------------------------
# Auto-detection by scanning the project directory
# ---------------------------------------------------------------------------

# Directories to skip during the file scan
_SKIP_DIRS: set[str] = {
    ".git", "node_modules", "__pycache__", "build", "dist", "target",
    ".dart_tool", ".next", "venv", ".venv", ".idea", ".vs", "vendor",
    "cmake-build-debug", "cmake-build-release", ".cache", ".gradle",
    "Pods", ".build", "egg-info",
}

# All known code extensions mapped to their language
_EXT_TO_LANG: dict[str, str] = {
    ".dart": "dart",
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hxx": "cpp", ".hh": "cpp",
}

# Max files to scan before stopping (avoid huge repos taking forever)
_MAX_SCAN_FILES: int = 5000


def detect_language(codebase_path: Path) -> str:
    """Auto-detect the primary language by scanning files in the project.

    Walks the directory tree (skipping common non-source dirs), counts code
    file extensions, and returns the language with the most source files.
    Falls back to "generic" if no known code files are found.
    """
    counts: Counter[str] = Counter()
    scanned = 0

    for root, dirs, files in os.walk(codebase_path):
        # Prune directories we don't want to descend into
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            lang = _EXT_TO_LANG.get(ext)
            if lang:
                counts[lang] += 1

            scanned += 1
            if scanned >= _MAX_SCAN_FILES:
                break
        if scanned >= _MAX_SCAN_FILES:
            break

    if not counts:
        return "generic"

    # .h files are ambiguous — could be C or C++. If we have .cpp/.cc files,
    # count .h towards C++. Otherwise count towards C.
    # (This is already handled by the counter — .h maps to "c" by default,
    #  but if cpp count > c count the user likely has a C++ project.)

    # Merge typescript into javascript (same profile)
    if "typescript" in counts:
        counts["javascript"] += counts.pop("typescript")

    winner = counts.most_common(1)[0][0]

    print(f"[detect] Scanned {scanned} files — language breakdown: {dict(counts.most_common())}")

    return winner


def get_profile(codebase_path: Path, language_override: str | None = None) -> LanguageProfile | None:
    """Get the language profile for a codebase.

    Args:
        codebase_path: Root of the target codebase.
        language_override: If set, use this language instead of auto-detecting.

    Returns:
        A LanguageProfile, or None if the language is "generic" (no profile).
    """
    lang = language_override or detect_language(codebase_path)
    return LANGUAGE_PROFILES.get(lang)
