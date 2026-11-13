"""Sandbox for the research agent's generated Python search plans.

The plan is untrusted input (LLM output). Layers, in order:

1. Static check (AST): only an allow-list of syntax. No imports, function/class
   definitions, while loops, global/nonlocal, with/await/yield/raise/try, and no
   name or attribute starting with "_" (the classic escape route via __class__,
   __globals__, __subclasses__ ...). `.format` / `.format_map` are blocked because
   format strings can reach those attributes too.
2. Allow-listed builtins only – no open, eval, exec, getattr, __import__, type, vars.
3. Execution budget enforced while running: a line counter and a wall-clock limit
   (checked by a trace function, so the plan is actually stopped, not abandoned);
   range() is capped.
4. Output: the plan must assign `result`; it is converted to JSON and capped.
   print() output is captured and capped.

The only way to reach data is the clearance-bound Catalog (app/rlm/catalog.py).

Limits of this design: it is defence in depth inside the API process, not a hard
security boundary. Production would run plans in a separate, resource-limited
process or container (see decision A16).
"""

import ast
import asyncio
import io
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from app.rlm.catalog import Catalog, CatalogError

MAX_CODE_CHARS = 4000
MAX_LINES_EXECUTED = 20_000
MAX_SECONDS = 20.0  # includes search() network time; the line budget stops runaway loops
MAX_RANGE = 10_000
MAX_INT_LITERAL = 100_000
MAX_RESULT_CHARS = 12_000
MAX_PRINT_CHARS = 2_000
SEARCH_TIMEOUT = 15.0

PLAN_FILENAME = "<research-plan>"

ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.If, ast.Break, ast.Continue,
    ast.Pass, ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.Subscript, ast.Slice,
    ast.Attribute, ast.Call, ast.keyword, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.JoinedStr, ast.FormattedValue, ast.Lambda, ast.arguments, ast.arg, ast.Starred,
    # operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
)
BLOCKED_ATTRIBUTES = {"format", "format_map", "mro", "gi_frame", "f_globals", "f_locals", "co_code"}


class SandboxError(Exception):
    """The plan was rejected or failed. The message is safe to show to the model so it can fix its plan."""


@dataclass
class SandboxResult:
    ok: bool
    result: Any = None
    stdout: str = ""
    error: str | None = None
    lines_executed: int = 0
    seconds: float = 0.0
    catalog_usage: dict = field(default_factory=dict)

    def for_model(self) -> str:
        """Compact JSON the agent sees after running a plan."""
        if not self.ok:
            return json.dumps({"ok": False, "error": self.error})
        return json.dumps({"ok": True, "result": self.result, "printed": self.stdout or None}, default=str)


# ---------- 1. static check ----------

def validate(code: str) -> ast.Module:
    if len(code) > MAX_CODE_CHARS:
        raise SandboxError(f"plan is too long ({len(code)} chars, max {MAX_CODE_CHARS})")
    try:
        tree = ast.parse(code, filename=PLAN_FILENAME, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"syntax error on line {exc.lineno}: {exc.msg}") from None

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise SandboxError(f"'{type(node).__name__}' is not allowed in a plan (line {getattr(node, 'lineno', '?')})")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise SandboxError(f"names starting with '_' are not allowed: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in BLOCKED_ATTRIBUTES:
                raise SandboxError(f"attribute '{node.attr}' is not allowed")
        if isinstance(node, ast.arg) and node.arg.startswith("_"):
            raise SandboxError(f"names starting with '_' are not allowed: {node.arg}")
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and abs(node.value) > MAX_INT_LITERAL:
            raise SandboxError(f"integer literal too large: {node.value}")
        if isinstance(node, ast.Constant) and isinstance(node.value, (bytes,)):
            raise SandboxError("bytes literals are not allowed")
    return tree


# ---------- 2. allowed functions ----------

def _capped_range(*args):
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise SandboxError(f"range() larger than {MAX_RANGE}")
    return r


def _chunk(items, size: int = 3) -> list[list]:
    """Split a list into batches – used to fan out work to sub-agents."""
    size = int(size)
    if size < 1:
        raise SandboxError("chunk size must be >= 1")
    items = list(items)
    return [items[i:i + size] for i in range(0, len(items), size)]


SAFE_BUILTINS = {
    "len": len, "range": _capped_range, "min": min, "max": max, "sum": sum, "sorted": sorted,
    "reversed": reversed, "enumerate": enumerate, "zip": zip, "any": any, "all": all, "abs": abs,
    "round": round, "str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict,
    "set": set, "tuple": tuple, "isinstance": isinstance, "True": True, "False": False, "None": None,
}


def _catalog_functions(catalog: Catalog, loop: asyncio.AbstractEventLoop | None) -> dict:
    def search(query: str, top_k: int = 5, document_type: str | None = None,
               department: str | None = None, since: str | None = None) -> list[dict]:
        if loop is None:
            raise SandboxError("search() is not available in this context")
        # The plan runs in a worker thread; search is async and runs on the API's event loop.
        future = asyncio.run_coroutine_threadsafe(
            catalog.search(query, top_k=top_k, document_type=document_type, department=department, since=since),
            loop)
        return future.result(timeout=SEARCH_TIMEOUT)

    return {
        "describe": catalog.describe,
        "list_docs": catalog.list_docs,
        "get_doc": catalog.get_doc,
        "read_section": catalog.read_section,
        "search": search,
        "chunk": _chunk,
    }


# ---------- 3. run with a budget ----------

class _Budget:
    def __init__(self, max_lines: int, max_seconds: float):
        self.max_lines, self.deadline, self.lines = max_lines, time.monotonic() + max_seconds, 0

    def trace(self, frame, event, _arg):
        # Count only lines of the plan itself (not catalog / library code).
        if frame.f_code.co_filename != PLAN_FILENAME:
            return None
        if event == "line":
            self.lines += 1
            if self.lines > self.max_lines:
                raise SandboxError(f"plan stopped: more than {self.max_lines} lines executed")
            if time.monotonic() > self.deadline:
                raise SandboxError("plan stopped: time limit reached")
        return self.trace


def _run_sync(tree: ast.Module, namespace: dict, budget: _Budget) -> None:
    code = compile(tree, PLAN_FILENAME, "exec")
    sys.settrace(budget.trace)  # this worker thread only; other threads are not affected
    try:
        exec(code, namespace)  # noqa: S102 – validated AST, restricted builtins and budget (see module doc)
    finally:
        sys.settrace(None)


def _jsonable(value: Any) -> Any:
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError) as exc:
        raise SandboxError(f"result is not JSON-serialisable: {exc}") from None
    if len(text) > MAX_RESULT_CHARS:
        raise SandboxError(f"result too large ({len(text)} chars, max {MAX_RESULT_CHARS}); "
                           "return ids and short fields, then read sections in a later step")
    return json.loads(text)


async def run_plan(code: str, catalog: Catalog, max_seconds: float = MAX_SECONDS,
                   max_lines: int = MAX_LINES_EXECUTED) -> SandboxResult:
    """Validate and run a plan. Never raises for plan problems – returns ok=False with a fixable message."""
    start = time.perf_counter()
    printed = io.StringIO()

    def _print(*args, **_kw):
        if printed.tell() < MAX_PRINT_CHARS:
            printed.write(" ".join(str(a) for a in args)[:MAX_PRINT_CHARS] + "\n")

    namespace: dict = {"__builtins__": {**SAFE_BUILTINS, "print": _print}}
    budget = _Budget(max_lines, max_seconds)
    try:
        tree = validate(code)
        loop = asyncio.get_running_loop()
        namespace.update(_catalog_functions(catalog, loop))
        await asyncio.wait_for(asyncio.to_thread(_run_sync, tree, namespace, budget), timeout=max_seconds + 2)
        if "result" not in namespace:
            raise SandboxError("the plan must assign its output to a variable named `result`")
        result = _jsonable(namespace["result"])
        return SandboxResult(True, result, printed.getvalue()[:MAX_PRINT_CHARS], None, budget.lines,
                             round(time.perf_counter() - start, 3), dict(catalog.usage))
    except (SandboxError, CatalogError) as exc:
        error = str(exc)
    except asyncio.TimeoutError:
        error = "plan stopped: time limit reached"
    except Exception as exc:  # noqa: BLE001 – any runtime error in generated code is reported back to the model
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    return SandboxResult(False, None, printed.getvalue()[:MAX_PRINT_CHARS], error, budget.lines,
                         round(time.perf_counter() - start, 3), dict(catalog.usage))
