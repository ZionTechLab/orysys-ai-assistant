"""Sandbox for the research agent's generated Python search plans.

The plan is untrusted input (LLM output). Layers, in order:

1. Static check (AST): only an allow-list of syntax. No imports, function/class
   definitions, while loops, global/nonlocal, with/await/yield/raise/try, and no
   name or attribute starting with "_" (the classic escape route via __class__,
   __globals__, __subclasses__ ...). `.format` / `.format_map` are blocked because
   format strings can reach those attributes too.
2. Allow-listed builtins only – no open, eval, exec, getattr, __import__, type, vars.
3. Execution budget enforced while running: a line counter, a wall-clock limit and a
   memory limit (checked by a trace function on every line – comprehension iterations
   count too – so the plan is actually stopped, not abandoned); range() is capped.
4. Memory blow-ups in a single step are refused before they allocate: `*` and `+` on
   strings / lists / tuples, `+=` / `*=`, str.join and str.replace are size-checked;
   padding methods (ljust, center, zfill ...) and wide format specs are blocked;
   sum() accepts numbers only. Gradual growth (e.g. list.extend in a loop) is caught
   by the memory limit.
5. Output: the plan must assign `result`; it is converted to JSON and capped.
   print() output is captured and capped.

The only way to reach data is the clearance-bound Catalog (app/rlm/catalog.py).

Limits of this design: it is defence in depth inside the API process, not a hard
security boundary (the memory check measures the whole process). Production would
run plans in a separate process or container with OS-level CPU / memory limits
(see decision A5).
"""

import ast
import asyncio
import copy
import io
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

from app.rlm.catalog import Catalog, CatalogError

MAX_CODE_CHARS = 6000  # analysis code is longer than search plans
MAX_LINES_EXECUTED = 20_000
MAX_SECONDS = 20.0  # includes search() network time; the line budget stops runaway loops
MAX_RANGE = 10_000
MAX_INT_LITERAL = 100_000
MAX_RESULT_CHARS = 12_000
MAX_PRINT_CHARS = 2_000
SEARCH_TIMEOUT = 15.0
MAX_SEQUENCE = 1_000_000      # characters / items one operation may produce
MAX_INT_RESULT = 10**18       # integers beyond this are not document analysis
MAX_MEMORY_MB = 256           # growth of the process while the plan runs
MEMORY_CHECK_EVERY = 10       # lines
MAX_FORMAT_WIDTH = 1000       # f"{x:>1000000}" would allocate a million characters

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
BLOCKED_ATTRIBUTES = {
    "format", "format_map", "mro",
    # frame / code objects of generators and coroutines
    "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code", "f_globals", "f_locals", "f_back",
    "f_builtins", "tb_frame", "co_code", "co_consts",
    # padding methods: one call can allocate an arbitrary amount of memory
    "ljust", "rjust", "center", "zfill", "expandtabs",
}


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
        if isinstance(node, ast.Name) and node.id.startswith("_") and node.id != "_":  # bare "_" = throwaway
            raise SandboxError(f"names starting with '_' are not allowed: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in BLOCKED_ATTRIBUTES:
                raise SandboxError(f"attribute '{node.attr}' is not allowed")
        if isinstance(node, ast.arg) and node.arg.startswith("_") and node.arg != "_":
            raise SandboxError(f"names starting with '_' are not allowed: {node.arg}")
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and abs(node.value) > MAX_INT_LITERAL:
            raise SandboxError(f"integer literal too large: {node.value}")
        if isinstance(node, ast.Constant) and isinstance(node.value, (bytes,)):
            raise SandboxError("bytes literals are not allowed")
        if isinstance(node, ast.FormattedValue) and node.format_spec is not None:
            _check_format_spec(node.format_spec)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant) \
                and isinstance(node.left.value, str):
            raise SandboxError("'%' string formatting is not allowed; use f-strings")
    return tree


def _check_format_spec(spec: ast.AST) -> None:
    """f-string format specs must be constant and narrow (".2f" is fine, ">1000000" is not)."""
    for part in ast.walk(spec):
        if isinstance(part, ast.FormattedValue):
            raise SandboxError("computed format specs (f\"{x:{width}}\") are not allowed")
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            widths = [int(n) for n in re.findall(r"\d+", part.value)]
            if any(w > MAX_FORMAT_WIDTH for w in widths):
                raise SandboxError(f"format width larger than {MAX_FORMAT_WIDTH} is not allowed")


# ---------- size-checked operations (inserted into the plan by _GuardOperations) ----------

_SEQUENCES = (str, list, tuple)


def _too_big(n: int) -> SandboxError:
    return SandboxError(f"operation would create {n:,} items/characters (max {MAX_SEQUENCE:,}); "
                        "work with ids and short fields instead")


def _checked_mul(a, b):
    for seq, n in ((a, b), (b, a)):
        if isinstance(seq, _SEQUENCES) and isinstance(n, int) and len(seq) * max(n, 0) > MAX_SEQUENCE:
            raise _too_big(len(seq) * n)
    result = a * b
    if isinstance(result, int) and abs(result) > MAX_INT_RESULT:
        raise SandboxError("number too large")
    return result


def _checked_add(a, b):
    if isinstance(a, _SEQUENCES) and isinstance(b, _SEQUENCES) and len(a) + len(b) > MAX_SEQUENCE:
        raise _too_big(len(a) + len(b))
    result = a + b
    if isinstance(result, int) and abs(result) > MAX_INT_RESULT:
        raise SandboxError("number too large")
    return result


def _checked_method(obj, name: str, *args, **kwargs):
    """str.join / str.replace with the result size checked before it is built."""
    method = getattr(obj, name)
    if isinstance(obj, str) and name == "join" and args:
        items = list(args[0])
        size = sum(len(i) for i in items if isinstance(i, str)) + len(obj) * max(len(items) - 1, 0)
        if size > MAX_SEQUENCE:
            raise _too_big(size)
        return method(items, *args[1:], **kwargs)
    if isinstance(obj, str) and name == "replace" and len(args) >= 2 \
            and isinstance(args[0], str) and isinstance(args[1], str):
        occurrences = obj.count(args[0]) if args[0] else len(obj) + 1
        size = len(obj) + occurrences * max(len(args[1]) - len(args[0]), 0)
        if size > MAX_SEQUENCE:
            raise _too_big(size)
    return method(*args, **kwargs)


def _checked_mod(a, b):
    if isinstance(a, str):  # "%1000000d" % 1 – printf-style widths allocate before any check
        raise SandboxError("'%' string formatting is not allowed; use f-strings")
    return a % b


def _numbers_sum(values, start=0):
    """sum() for numbers only: sum(list_of_lists, []) could build a huge list in one call."""
    if not isinstance(start, (int, float)):
        raise SandboxError("sum() works on numbers only")
    return sum(values, start)


_GUARDED_METHODS = {"join", "replace"}
_OP_HELPERS = {ast.Mult: "__sb_mul", ast.Add: "__sb_add", ast.Mod: "__sb_mod"}
_HELPERS = {"__sb_mul": _checked_mul, "__sb_add": _checked_add, "__sb_mod": _checked_mod,
            "__sb_method": _checked_method}


class _GuardOperations(ast.NodeTransformer):
    """Rewrite a validated plan so size-risky operations go through the checked helpers.

    Runs after validate(), so the plan itself still cannot name these helpers (names starting
    with "_" are rejected).
    """

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        helper = _OP_HELPERS.get(type(node.op))
        if helper is None:
            return node
        return ast.copy_location(ast.Call(ast.Name(helper, ast.Load()), [node.left, node.right], []), node)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        helper = _OP_HELPERS.get(type(node.op))
        if helper is None:
            return node
        current = copy.deepcopy(node.target)
        current.ctx = ast.Load()
        value = ast.Call(ast.Name(helper, ast.Load()), [current, node.value], [])
        return ast.copy_location(ast.Assign([node.target], value), node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _GUARDED_METHODS:
            args = [func.value, ast.Constant(func.attr), *node.args]
            return ast.copy_location(ast.Call(ast.Name("__sb_method", ast.Load()), args, node.keywords), node)
        return node


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
    "len": len, "range": _capped_range, "min": min, "max": max, "sum": _numbers_sum, "sorted": sorted,
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
    def __init__(self, max_lines: int, max_seconds: float, max_memory_mb: float = MAX_MEMORY_MB):
        self.max_lines, self.deadline, self.lines = max_lines, time.monotonic() + max_seconds, 0
        self._process = psutil.Process()
        self.max_growth = max_memory_mb * 1024 * 1024
        self.baseline = self._process.memory_info().rss
        self.peak_growth = 0

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
            if self.lines % MEMORY_CHECK_EVERY == 0:
                self.check_memory()
        return self.trace

    def check_memory(self) -> None:
        growth = self._process.memory_info().rss - self.baseline
        self.peak_growth = max(self.peak_growth, growth)
        if growth > self.max_growth:
            raise SandboxError(f"plan stopped: memory limit reached ({self.max_growth // (1024 * 1024)} MB)")


def _run_sync(tree: ast.Module, namespace: dict, budget: _Budget) -> None:
    tree = ast.fix_missing_locations(_GuardOperations().visit(tree))
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
                   max_lines: int = MAX_LINES_EXECUTED, extra_functions: dict | None = None,
                   max_memory_mb: float = MAX_MEMORY_MB) -> SandboxResult:
    """Validate and run a plan. Never raises for plan problems – returns ok=False with a fixable message.

    extra_functions: additional safe helpers (e.g. analysis functions); names must not start with "_".
    """
    start = time.perf_counter()
    printed = io.StringIO()

    def _print(*args, **_kw):
        if printed.tell() < MAX_PRINT_CHARS:
            printed.write(" ".join(str(a) for a in args)[:MAX_PRINT_CHARS] + "\n")

    namespace: dict = {"__builtins__": {**SAFE_BUILTINS, "print": _print}, **_HELPERS}
    budget = _Budget(max_lines, max_seconds, max_memory_mb)
    try:
        tree = validate(code)
        loop = asyncio.get_running_loop()
        namespace.update(_catalog_functions(catalog, loop))
        for name, fn in (extra_functions or {}).items():
            if name.startswith("_") or name in namespace:
                raise ValueError(f"invalid extra function name: {name}")
            namespace[name] = fn
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
