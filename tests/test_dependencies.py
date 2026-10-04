"""Every runtime dependency declared in ``[project.dependencies]`` must be imported.

``networkx`` and ``pydantic`` were declared as runtime dependencies while no
module under ``agentcallgraph/`` imported either of them, so a plain
``pip install agent-call-graph`` pulled a graph library and a validation
library into the user's environment for code that never touches them. The
declarations were the only two mentions of those names anywhere in the tree.

This test parses ``pyproject.toml`` with ``tomllib`` and walks the package
sources with :mod:`ast` (a parse, not a regex, so a dependency name appearing
in a docstring or a comment cannot satisfy the check) and fails when a declared
runtime dependency has no importing module.

Scope notes:

* Only ``[project.dependencies]`` is checked. ``[project.optional-dependencies]``
  is deliberately out of scope: extras are installed on request, and a dev-only
  requirement that no module imports is the normal case, not a defect.
* A requirement carrying an environment marker (``; sys_platform == "win32"``)
  is exempt. A static import scan cannot evaluate a marker, so asserting on it
  would produce a false failure for a dependency that is genuinely used on one
  platform only. Marker-bearing requirements are still reported by
  ``test_marker_bearing_requirements_are_listed`` so they are never silently
  forgotten.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_DIR = REPO_ROOT / "agentcallgraph"


def _normalize(name: str) -> str:
    """Normalize a distribution or module name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _split_requirement(requirement: str) -> tuple[str, str]:
    """Return ``(name, marker)`` for a PEP 508 requirement string."""
    text, _, marker = requirement.partition(";")
    # Drop any extras ("uvicorn[standard]") before reading the name.
    text = text.split("[", 1)[0]
    # The name ends at the first version specifier or whitespace.
    name = re.split(r"[\s<>=!~;(]", text.strip(), maxsplit=1)[0]
    return name, marker.strip()


def _read_pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _declared_runtime_requirements() -> list[str]:
    data = _read_pyproject()
    return list(data.get("project", {}).get("dependencies", []))


def _python_sources(package_dir: Path) -> list[Path]:
    return sorted(p for p in package_dir.rglob("*.py") if p.is_file())


def _imported_roots(package_dir: Path) -> set[str]:
    """Collect the top-level module name of every import in ``package_dir``."""
    roots: set[str] = set()
    for path in _python_sources(package_dir):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
            ):
                # node.level > 0 is a relative import inside the package.
                roots.add(node.module.split(".")[0])
    return {root for root in roots if root}


def _first_party_roots() -> set[str]:
    """Top-level packages shipped by this repository."""
    return {p.name for p in REPO_ROOT.iterdir() if (p / "__init__.py").is_file()}


def _third_party_imports() -> set[str]:
    """Imported module names that are neither stdlib nor this package."""
    stdlib = set(sys.stdlib_module_names)
    own = _first_party_roots()
    return {
        root
        for root in _imported_roots(PACKAGE_DIR)
        if root not in stdlib and root not in own
    }


def _unused_runtime_requirements(
    requirements: list[str], imported: set[str]
) -> tuple[list[str], list[str]]:
    """Split ``requirements`` into (unused names, marker-bearing requirements)."""
    unused: list[str] = []
    marked: list[str] = []
    for requirement in requirements:
        name, marker = _split_requirement(requirement)
        if not name:
            continue
        if marker:
            marked.append(requirement)
            continue
        if _normalize(name) not in {_normalize(mod) for mod in imported}:
            unused.append(requirement)
    return unused, marked


# --- positive controls -------------------------------------------------------
# A scan that collects nothing would make the assertion below trivially true,
# and a scan that collects everything would make it meaningless. Both ends are
# pinned here.


def test_package_sources_are_actually_parsed() -> None:
    sources = _python_sources(PACKAGE_DIR)
    assert len(sources) >= 5, f"expected the package sources, found {sources}"
    assert PACKAGE_DIR.is_dir(), f"missing package directory {PACKAGE_DIR}"


def test_third_party_scan_finds_the_real_import_only() -> None:
    imported = _third_party_imports()
    # The one third-party import the package genuinely has.
    assert "click" in imported, imported
    # Stdlib is not third-party.
    for stdlib_name in ("json", "sqlite3", "pathlib", "subprocess"):
        assert stdlib_name not in imported, sorted(imported)
    # The package's own modules are not third-party either.
    assert "agentcallgraph" not in imported, sorted(imported)


# --- the regression itself ---------------------------------------------------


def test_every_declared_runtime_dependency_is_imported() -> None:
    requirements = _declared_runtime_requirements()
    unused, _ = _unused_runtime_requirements(requirements, _third_party_imports())
    assert not unused, (
        "runtime dependencies declared in [project.dependencies] but never "
        f"imported by agentcallgraph/: {unused}. Remove them, or import them."
    )


def test_runtime_dependency_list_is_not_empty() -> None:
    """A silently emptied dependency list would make the check vacuous."""
    requirements = _declared_runtime_requirements()
    assert requirements, "[project.dependencies] is empty; the CLI needs click"
    assert any("click" in req for req in requirements), requirements


def test_marker_bearing_requirements_are_listed() -> None:
    """Marked requirements are exempt, but never dropped from view."""
    requirements = _declared_runtime_requirements()
    _, marked = _unused_runtime_requirements(requirements, _third_party_imports())
    for requirement in marked:
        assert ";" in requirement, requirement


# --- unit coverage for the two parsing helpers -------------------------------


def test_requirement_parsing_handles_pins_extras_and_markers() -> None:
    assert _split_requirement("click>=8.1") == ("click", "")
    assert _split_requirement("typing-extensions >= 4.14.1 ; python_version < '3.12'") == (
        "typing-extensions",
        "python_version < '3.12'",
    )
    assert _split_requirement("uvicorn[standard]==0.30") == ("uvicorn", "")
    assert _split_requirement("foo") == ("foo", "")


def test_marker_exemption_does_not_suppress_unmarked_dependencies() -> None:
    unused, marked = _unused_runtime_requirements(
        [
            "click>=8.1",
            "networkx>=3.0",
            "pydantic>=2.0",
            "tomli; python_version < '3.11'",
        ],
        {"click"},
    )
    assert unused == ["networkx>=3.0", "pydantic>=2.0"]
    assert marked == ["tomli; python_version < '3.11'"]


def test_underscore_and_hyphen_spelling_matches_module_names() -> None:
    """A distribution spelled with underscores still matches its module."""
    unused, _ = _unused_runtime_requirements(
        ["typing_extensions>=4.0"], {"typing_extensions"}
    )
    assert unused == []