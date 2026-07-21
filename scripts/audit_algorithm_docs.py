#!/usr/bin/env python3
"""Audit ReactFlow algorithm implementation documentation.

The user-facing requirement is stronger than "tests pass": public algorithms
must not contain pseudocode/placeholder implementations, and their docstrings
must explain implementation logic with complexity and, where relevant, formulas.

This script turns that requirement into a repeatable AST audit.  It does not
prove scientific correctness by itself; instead, it creates the gap list needed
for a paper-grade completion audit.

Complexity: O(F + S), where F is the number of parsed Python files and S is the
total source size.  The AST walk is linear in source nodes.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


COMPLEXITY_MARKERS = ("Complexity", "复杂度")
MATH_MARKERS = (
    "Formula",
    "Mathematical",
    "math",
    "gradient",
    "loss",
    "posterior",
    "probability",
    "公式",
    "L_",
    "O(",
)
ALGORITHM_NAME_RE = re.compile(
    r"(loss|gradient|forward|backward|sample|project|normalize|validate|split|"
    r"cluster|weight|score|metric|matrix|prob|energy|partition|denois|reactiv|"
    r"feature|adapter|audit|monitor)",
    re.IGNORECASE,
)
TEXT_PLACEHOLDER_RE = re.compile(
    r"\b(TODO|FIXME|NotImplemented|placeholder|pseudo[- ]?code)\b|伪代码|占位",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DocAuditRow:
    """One public AST node documentation audit row.

    Complexity: O(1) storage per row.
    """

    path: str
    line: int
    kind: str
    qualified_name: str
    has_docstring: bool
    has_complexity: bool
    requires_math: bool
    has_math_marker: bool
    status: str
    detail: str


@dataclass(frozen=True)
class PlaceholderRow:
    """One placeholder or pseudocode finding.

    Complexity: O(1) storage per row.
    """

    path: str
    line: int
    kind: str
    qualified_name: str
    detail: str


def _is_public_name(name: str) -> bool:
    """Return whether an AST symbol is part of the public audit surface.

    Complexity: O(1).
    """

    return bool(name) and not name.startswith("_")


def _has_marker(text: str, markers: Sequence[str]) -> bool:
    """Return whether ``text`` contains any marker.

    Complexity: O(M * T), where M is the number of markers and T is text length.
    """

    return any(marker in text for marker in markers)


def _decorator_names(node: ast.AST) -> Tuple[str, ...]:
    """Return normalized decorator names for a class/function node.

    Complexity: O(D), where D is the number of decorators.
    """

    decorators = getattr(node, "decorator_list", ())
    names: List[str] = []
    for decorator in decorators:
        if isinstance(decorator, ast.Name):
            names.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            names.append(decorator.attr)
        elif isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return tuple(names)


def _requires_math(kind: str, qualified_name: str, decorators: Sequence[str]) -> bool:
    """Return whether a node should include a formula/math marker.

    Dataclasses and simple containers are documentation targets, but they usually
    do not need a formula.  Algorithmic names such as ``loss`` or ``gradient`` do.

    Complexity: O(len(qualified_name)).
    """

    if kind == "class" and "dataclass" in decorators:
        return False
    return bool(ALGORITHM_NAME_RE.search(qualified_name))


def _placeholder_from_body(path: Path, node: ast.AST, kind: str, qualified_name: str) -> Iterator[PlaceholderRow]:
    """Yield placeholder rows from a single class/function body.

    The audit treats bodies that consist only of ``pass`` or ``...`` as hard
    placeholder risks.  Incidental ``pass`` in an exception handler is ignored
    because it can be a deliberate no-op.

    Complexity: O(B), where B is the immediate body length.
    """

    body = list(getattr(node, "body", ()))
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if len(body) == 1 and isinstance(body[0], ast.Pass):
        yield PlaceholderRow(str(path), body[0].lineno, kind, qualified_name, "body contains only pass")
    if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if body[0].value.value is Ellipsis:
            yield PlaceholderRow(str(path), body[0].lineno, kind, qualified_name, "body contains only ellipsis")
    for child in ast.walk(node):
        if isinstance(child, ast.Raise):
            exc = child.exc
            if isinstance(exc, ast.Call):
                exc = exc.func
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                yield PlaceholderRow(str(path), child.lineno, kind, qualified_name, "raises NotImplementedError")


class AlgorithmDocVisitor(ast.NodeVisitor):
    """AST visitor collecting public doc rows and placeholder findings.

    Complexity: O(N), where N is the number of AST nodes.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.scope: List[str] = []
        self.rows: List[DocAuditRow] = []
        self.placeholders: List[PlaceholderRow] = []

    def _qualified(self, name: str) -> str:
        """Return the current qualified name.

        Complexity: O(depth).
        """

        return ".".join([*self.scope, name])

    def _record_doc_node(self, node: ast.AST, *, kind: str, name: str) -> None:
        """Record one public class/function/method documentation row.

        Complexity: O(docstring length).
        """

        decorators = _decorator_names(node)
        qualified_name = self._qualified(name)
        doc = ast.get_docstring(node) or ""
        has_doc = bool(doc)
        has_complexity = _has_marker(doc, COMPLEXITY_MARKERS)
        requires_math = _requires_math(kind, qualified_name, decorators)
        has_math = _has_marker(doc, MATH_MARKERS)

        problems: List[str] = []
        if not has_doc:
            problems.append("missing docstring")
        if has_doc and not has_complexity:
            problems.append("missing complexity")
        if requires_math and has_doc and not has_math:
            problems.append("missing math/formula marker")
        status = "pass" if not problems else "warn"

        self.rows.append(
            DocAuditRow(
                path=str(self.path),
                line=getattr(node, "lineno", 0),
                kind=kind,
                qualified_name=qualified_name,
                has_docstring=has_doc,
                has_complexity=has_complexity,
                requires_math=requires_math,
                has_math_marker=has_math,
                status=status,
                detail="; ".join(problems),
            )
        )
        self.placeholders.extend(_placeholder_from_body(self.path, node, kind, qualified_name))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 - ast API
        if _is_public_name(node.name):
            self._record_doc_node(node, kind="class", name=node.name)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast API
        if _is_public_name(node.name):
            kind = "method" if self.scope else "function"
            self._record_doc_node(node, kind=kind, name=node.name)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802 - ast API
        if _is_public_name(node.name):
            kind = "async_method" if self.scope else "async_function"
            self._record_doc_node(node, kind=kind, name=node.name)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def iter_python_files(paths: Sequence[Path]) -> Iterator[Path]:
    """Yield Python files from explicit files or directories.

    Complexity: O(P + F log F), where P is input path count and F is discovered
    Python file count.
    """

    for path in paths:
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.py"))


def audit_file(path: Path) -> Tuple[List[DocAuditRow], List[PlaceholderRow], List[PlaceholderRow]]:
    """Audit one Python source file.

    The third returned list contains text-marker rows such as TODO/FIXME.

    Complexity: O(file size).
    """

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    visitor = AlgorithmDocVisitor(path)
    visitor.visit(tree)

    text_rows: List[PlaceholderRow] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = TEXT_PLACEHOLDER_RE.search(line)
        if match and "audit_algorithm_docs.py" not in str(path):
            text_rows.append(
                PlaceholderRow(str(path), lineno, "text", "<module>", f"marker={match.group(0)!r}")
            )
    return visitor.rows, visitor.placeholders, text_rows


def run_audit(paths: Sequence[Path]) -> dict:
    """Run the full algorithm documentation audit.

    Complexity: O(total source size).
    """

    rows: List[DocAuditRow] = []
    placeholders: List[PlaceholderRow] = []
    text_markers: List[PlaceholderRow] = []
    parse_errors: List[Mapping[str, object]] = []

    for path in iter_python_files(paths):
        try:
            file_rows, file_placeholders, file_text_markers = audit_file(path)
        except SyntaxError as exc:
            parse_errors.append({"path": str(path), "line": exc.lineno, "detail": str(exc)})
            continue
        rows.extend(file_rows)
        placeholders.extend(file_placeholders)
        text_markers.extend(file_text_markers)

    missing_doc = sum(not row.has_docstring for row in rows)
    missing_complexity = sum(row.has_docstring and not row.has_complexity for row in rows)
    missing_math = sum(row.requires_math and row.has_docstring and not row.has_math_marker for row in rows)
    summary = {
        "public_nodes": len(rows),
        "passing_doc_rows": sum(row.status == "pass" for row in rows),
        "missing_docstrings": missing_doc,
        "missing_complexity": missing_complexity,
        "missing_math_markers": missing_math,
        "placeholder_bodies": len(placeholders),
        "text_markers": len(text_markers),
        "parse_errors": len(parse_errors),
        "strict_ready": (
            missing_doc == 0
            and missing_complexity == 0
            and missing_math == 0
            and not placeholders
            and not text_markers
            and not parse_errors
        ),
    }
    return {
        "summary": summary,
        "rows": [asdict(row) for row in rows],
        "placeholder_rows": [asdict(row) for row in placeholders],
        "text_marker_rows": [asdict(row) for row in text_markers],
        "parse_errors": parse_errors,
    }


def _md_escape(text: object) -> str:
    """Escape table-sensitive Markdown characters.

    Complexity: O(len(text)).
    """

    return str(text).replace("|", "\\|").replace("\n", " ")


def write_markdown(result: Mapping[str, object], path: Path, *, max_rows: int = 200) -> None:
    """Write audit result as Markdown.

    Complexity: O(N), where N is the number of emitted rows.
    """

    summary = result["summary"]
    rows = result["rows"]
    placeholders = result["placeholder_rows"]
    text_markers = result["text_marker_rows"]
    parse_errors = result["parse_errors"]

    lines = [
        "# ReactFlow Algorithm Documentation Audit",
        "",
        f"- summary: `{summary}`",
        "",
        "## Placeholder / Pseudocode Findings",
        "",
    ]
    if placeholders or text_markers or parse_errors:
        lines.extend(["| Kind | Path | Line | Qualified Name | Detail |", "|---|---|---:|---|---|"])
        for item in list(placeholders) + list(text_markers) + list(parse_errors):
            lines.append(
                "| {kind} | {path} | {line} | {name} | {detail} |".format(
                    kind=_md_escape(item.get("kind", "parse_error")),
                    path=_md_escape(item.get("path", "")),
                    line=_md_escape(item.get("line", "")),
                    name=_md_escape(item.get("qualified_name", "")),
                    detail=_md_escape(item.get("detail", "")),
                )
            )
    else:
        lines.append("No placeholder, pseudocode, or parse-error findings.")

    lines.extend(
        [
            "",
            "## Documentation Gaps",
            "",
            "| Status | Path | Line | Kind | Qualified Name | Detail |",
            "|---|---|---:|---|---|---|",
        ]
    )
    gap_rows = [row for row in rows if row["status"] != "pass"]
    for row in gap_rows[:max_rows]:
        lines.append(
            "| {status} | {path} | {line} | {kind} | {name} | {detail} |".format(
                status=_md_escape(row["status"]),
                path=_md_escape(row["path"]),
                line=_md_escape(row["line"]),
                kind=_md_escape(row["kind"]),
                name=_md_escape(row["qualified_name"]),
                detail=_md_escape(row["detail"]),
            )
        )
    if len(gap_rows) > max_rows:
        lines.append(f"| warn | ... |  |  |  | truncated {len(gap_rows) - max_rows} rows |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Complexity: O(total source size).
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["src/reactflow"], help="Python files or directories to audit")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--fail-on-placeholder", action="store_true")
    parser.add_argument("--fail-on-doc-gaps", action="store_true")
    args = parser.parse_args(argv)

    result = run_audit([Path(path) for path in args.paths])
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, Path(args.output_md))
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))

    summary = result["summary"]
    if args.fail_on_placeholder and (
        summary["placeholder_bodies"] or summary["text_markers"] or summary["parse_errors"]
    ):
        return 1
    if args.fail_on_doc_gaps and not summary["strict_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
