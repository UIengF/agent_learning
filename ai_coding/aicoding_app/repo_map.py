from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "runtime"}
DEPENDENCY_FILES = {
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "setup.cfg",
    "poetry.lock",
    "uv.lock",
}


@dataclass(frozen=True)
class PythonSymbol:
    kind: str
    name: str
    path: str
    line: int


@dataclass(frozen=True)
class PythonImports:
    path: str
    imports: tuple[str, ...]


@dataclass(frozen=True)
class RepoMap:
    files: tuple[str, ...] = ()
    test_roots: tuple[str, ...] = ()
    dependency_files: tuple[str, ...] = ()
    python_symbols: tuple[PythonSymbol, ...] = ()
    python_imports: tuple[PythonImports, ...] = ()
    parse_errors: tuple[str, ...] = ()

    def module_names_for_path(self, path: str) -> tuple[str, ...]:
        normalized = path.replace("\\", "/").removesuffix(".py")
        parts = [part for part in normalized.split("/") if part]
        if not parts:
            return ()
        names = [".".join(parts)]
        if parts[-1] == "__init__":
            names.append(".".join(parts[:-1]))
        names.append(parts[-1])
        return tuple(item for item in dict.fromkeys(names) if item)

    def imports_for_path(self, path: str) -> tuple[str, ...]:
        normalized = path.replace("\\", "/")
        for item in self.python_imports:
            if item.path == normalized:
                return item.imports
        return ()

    def reverse_imports_for_path(self, path: str) -> tuple[str, ...]:
        module_names = self.module_names_for_path(path)
        importers: list[str] = []
        for item in self.python_imports:
            if item.path == path:
                continue
            for imported in item.imports:
                if _import_matches_module(imported, module_names):
                    importers.append(item.path)
                    break
        return tuple(dict.fromkeys(importers))

    def symbols_named(self, query: str) -> tuple[PythonSymbol, ...]:
        lowered = query.lower()
        exact = [item for item in self.python_symbols if item.name == query]
        if exact:
            return tuple(exact)
        return tuple(item for item in self.python_symbols if item.name.lower() == lowered)

    def related_files_for_path(self, path: str) -> tuple[str, ...]:
        normalized = path.replace("\\", "/")
        if normalized not in self.files:
            return ()
        candidates: list[str] = []
        stem = Path(normalized).stem
        if stem.startswith("test_"):
            source_stem = stem.removeprefix("test_")
            candidates.extend(
                item
                for item in self.files
                if item.endswith(".py")
                and not _is_test_file(item)
                and Path(item).stem in {source_stem, source_stem.removesuffix("_test")}
            )
        else:
            candidates.extend(
                item
                for item in self.files
                if item.endswith(".py")
                and _is_test_file(item)
                and Path(item).stem in {f"test_{stem}", f"{stem}_test"}
            )
            candidates.extend(self.reverse_imports_for_path(normalized))
        return tuple(dict.fromkeys(item for item in candidates if item != normalized))

    def explain(self, query: str) -> str:
        stripped = query.strip()
        normalized = stripped.replace("\\", "/")
        if normalized in self.files:
            return self._explain_file(normalized)

        symbols = self.symbols_named(stripped)
        if symbols:
            lines = [f"Query: {stripped}", "Symbol definitions:"]
            lines.extend(f"- {item.path}:{item.line}: {item.kind} {item.name}" for item in symbols)
            related: list[str] = []
            for item in symbols:
                related.extend(self.related_files_for_path(item.path))
                related.extend(self.reverse_imports_for_path(item.path))
            lines.extend(["Related files:", _format_items(tuple(dict.fromkeys(related)))])
            return "\n".join(lines)

        fuzzy_files = [item for item in self.files if stripped.lower() in item.lower()]
        if fuzzy_files:
            lines = [f"Query: {stripped}", "Matching files:"]
            lines.extend(f"- {item}" for item in fuzzy_files[:20])
            return "\n".join(lines)
        return "\n".join([f"Query: {stripped}", "No matching symbol or file found."])

    def _explain_file(self, path: str) -> str:
        symbols = [item for item in self.python_symbols if item.path == path]
        imports = self.imports_for_path(path)
        reverse_imports = self.reverse_imports_for_path(path)
        related_files = self.related_files_for_path(path)
        lines = [
            f"File: {path}",
            "Symbols:",
            _format_items(tuple(f"{item.line}: {item.kind} {item.name}" for item in symbols)),
            "Imports:",
            _format_items(imports),
            "Reverse imports:",
            _format_items(reverse_imports),
            "Related files:",
            _format_items(related_files),
        ]
        return "\n".join(lines)

    def format(self, *, max_files: int = 80, max_symbols: int = 120) -> str:
        file_lines = list(self.files[:max_files])
        if len(self.files) > max_files:
            file_lines.append(f"... {len(self.files) - max_files} more files")
        symbol_lines = [
            f"{item.path}:{item.line}: {item.kind} {item.name}"
            for item in self.python_symbols[:max_symbols]
        ]
        if len(self.python_symbols) > max_symbols:
            symbol_lines.append(f"... {len(self.python_symbols) - max_symbols} more symbols")
        import_lines = [
            f"{item.path}: {', '.join(item.imports)}"
            for item in self.python_imports
            if item.imports
        ][:40]
        blocks = [
            "Files:",
            "\n".join(file_lines) or "none",
            "Test roots:",
            ", ".join(self.test_roots) or "none",
            "Dependency files:",
            ", ".join(self.dependency_files) or "none",
            "Python symbols:",
            "\n".join(symbol_lines) or "none",
            "Python imports:",
            "\n".join(import_lines) or "none",
        ]
        if self.parse_errors:
            blocks.extend(["Parse errors:", "\n".join(self.parse_errors[:20])])
        return "\n\n".join(blocks)


def _is_skipped(path: Path, workspace: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(workspace).parts)


def _relative(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _module_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        return ", ".join(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        prefix = "." * node.level
        return f"{prefix}{node.module or ''}"
    return None


def _import_matches_module(imported: str, module_names: tuple[str, ...]) -> bool:
    imported = imported.strip(".")
    if not imported:
        return False
    for module_name in module_names:
        if imported == module_name or imported.startswith(f"{module_name}."):
            return True
        if module_name.endswith(f".{imported}") or imported.endswith(f".{module_name}"):
            return True
    return False


def _is_test_file(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return "tests" in parts or name.startswith("test_") or name.endswith("_test.py")


def _format_items(items: tuple[str, ...]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def build_repo_map(workspace: str | Path, *, max_files: int = 500) -> RepoMap:
    root = Path(workspace).resolve()
    files: list[str] = []
    test_roots: set[str] = set()
    dependency_files: list[str] = []
    python_symbols: list[PythonSymbol] = []
    python_imports: list[PythonImports] = []
    parse_errors: list[str] = []

    for path in sorted(root.rglob("*")):
        if path.is_dir() or _is_skipped(path, root):
            continue
        rel = _relative(path, root)
        files.append(rel)
        if len(files) >= max_files:
            break
        parts = path.relative_to(root).parts
        if any(part == "tests" or part.startswith("test") for part in parts):
            test_roots.add(parts[0])
        if path.name in DEPENDENCY_FILES:
            dependency_files.append(rel)
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError) as exc:
            parse_errors.append(f"{rel}: {exc}")
            continue
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                python_symbols.append(PythonSymbol(kind=kind, name=node.name, path=rel, line=node.lineno))
            module = _module_name(node)
            if module:
                imports.append(module)
        python_imports.append(PythonImports(path=rel, imports=tuple(dict.fromkeys(imports))))

    return RepoMap(
        files=tuple(files),
        test_roots=tuple(sorted(test_roots)),
        dependency_files=tuple(dependency_files),
        python_symbols=tuple(python_symbols),
        python_imports=tuple(python_imports),
        parse_errors=tuple(parse_errors),
    )
