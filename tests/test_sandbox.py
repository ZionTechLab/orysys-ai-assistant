"""Sandbox for generated research plans: useful plans run, escapes and abuse are blocked."""

import pytest

from app.rlm.catalog import Catalog
from app.rlm.sandbox import run_plan

VIEWER = ("public", "internal")


@pytest.fixture
def catalog():
    return Catalog(VIEWER)


# ---------- plans that should work ----------

async def test_payment_outage_plan_runs(catalog):
    plan = """
docs = []
for dept in ["payments", "cards"]:
    docs += list_docs(document_type="incident", department=dept, since="2025-11-12")
ids = sorted(d["doc_id"] for d in docs)
batches = chunk(ids, 3)
result = {"count": len(ids), "batches": batches}
"""
    r = await run_plan(plan, catalog)
    assert r.ok, r.error
    assert r.result["count"] == 8
    assert [len(b) for b in r.result["batches"]] == [3, 3, 2]
    assert "INC-2025-101" not in sum(r.result["batches"], [])
    assert r.catalog_usage["list_docs"] == 2


async def test_plan_can_read_sections_and_filter(catalog):
    plan = """
causes = {}
for d in list_docs(document_type="incident", tag="paybridge"):
    text = read_section(d["doc_id"], "Root Cause")
    causes[d["doc_id"]] = "circuit breaker" in text.lower()
print("checked", len(causes))
result = causes
"""
    r = await run_plan(plan, catalog)
    assert r.ok, r.error
    assert r.result == {"INC-2026-095": True, "INC-2026-062": True, "INC-2026-021": True}
    assert r.stdout.strip() == "checked 3"
    assert r.catalog_usage["read_section"] == 3


async def test_comprehensions_lambdas_and_fstrings_are_allowed(catalog):
    plan = """
docs = sorted(list_docs(document_type="runbook"), key=lambda d: d["doc_id"])
result = [f"{d['doc_id']}: {len(d['sections'])} sections" for d in docs if d["department"] == "payments"]
"""
    r = await run_plan(plan, catalog)
    assert r.ok, r.error
    assert r.result[0].startswith("RB-002")


# ---------- plans that must be blocked before running ----------

@pytest.mark.parametrize("code,reason", [
    ("import os\nresult = os.listdir('.')", "Import"),
    ("from os import path\nresult = 1", "ImportFrom"),
    ("result = open('.env').read()", "open"),
    ("result = __import__('os')", "_"),
    ("result = ().__class__.__bases__[0].__subclasses__()", "__class__"),
    ("result = list_docs.__self__", "__self__"),
    ("result = '{0.__class__}'.format(1)", "format"),
    ("result = eval('1+1')", "eval"),
    ("result = getattr(describe, 'x')", "getattr"),
    ("def f():\n    return 1\nresult = f()", "FunctionDef"),
    ("class A:\n    pass\nresult = 1", "ClassDef"),
    ("while True:\n    pass", "While"),
    ("global x\nresult = 1", "Global"),
    ("try:\n    result = 1\nexcept Exception:\n    result = 2", "Try"),
    ("result = 10 ** 10", "Pow"),
    ("result = 'a' * 1000000", "too large"),
    ("_x = 1\nresult = _x", "_"),
    ("result = [x for x in range(10)]\nresult = result.gi_frame", "gi_frame"),
])
async def test_dangerous_plans_are_rejected(catalog, code, reason):
    r = await run_plan(code, catalog)
    assert not r.ok
    assert reason.lower() in r.error.lower() or "not allowed" in r.error.lower() or "not defined" in r.error.lower()
    assert r.lines_executed <= 1


# ---------- runtime limits ----------

async def test_line_budget_stops_long_loops(catalog):
    plan = "n = 0\nfor i in range(10000):\n    for j in range(10000):\n        n += 1\nresult = n"
    r = await run_plan(plan, catalog, max_lines=5000)
    assert not r.ok and "lines executed" in r.error


async def test_range_is_capped(catalog):
    r = await run_plan("result = len(range(1000 * 1000))", catalog)  # literal check alone is not enough
    assert not r.ok and "range" in r.error


async def test_time_budget_stops_plan(catalog):
    plan = "n = 0\nfor i in range(10000):\n    for j in range(10000):\n        n += 1\nresult = n"
    r = await run_plan(plan, catalog, max_seconds=0.3, max_lines=10**9)
    assert not r.ok and "time limit" in r.error


async def test_result_must_be_set(catalog):
    r = await run_plan("x = 1", catalog)
    assert not r.ok and "`result`" in r.error


async def test_large_results_are_refused_with_advice(catalog):
    plan = "result = [read_section(d['doc_id'], s) for d in list_docs() for s in d['sections']]"
    r = await run_plan(plan, catalog)
    assert not r.ok and "too large" in r.error and "read sections" in r.error


async def test_runtime_errors_are_reported_not_raised(catalog):
    r = await run_plan("result = list_docs()[0]['nope']", catalog)
    assert not r.ok and "KeyError" in r.error


# ---------- data access stays inside the user's clearance ----------

async def test_plan_cannot_read_restricted_documents(catalog):
    r = await run_plan("result = read_section('POL-007', 'Structure')", catalog)
    assert not r.ok and "not found" in r.error


async def test_plan_cannot_see_restricted_documents_in_listings(catalog):
    r = await run_plan("result = [d['doc_id'] for d in list_docs(document_type='policy')]", catalog)
    assert r.ok and "POL-007" not in r.result and "POL-008" not in r.result


async def test_error_message_helps_the_model_fix_the_plan(catalog):
    r = await run_plan("result = read_section('INC-2026-007', 'Lessons')", catalog)
    assert not r.ok and "Root Cause" in r.error  # lists the real section names
    assert '"ok": false' in r.for_model()


async def test_search_works_from_inside_a_plan():
    """The plan runs in a worker thread; search() is async and must run on the event loop."""
    from app.retrieval.hybrid import Hit

    class FakeRetriever:
        async def search(self, query, clearance, filters, top_k):
            return [Hit("RB-003#actions", "RB-003", "PayBridge", "Actions", "x", 0.5, "payments", "runbook",
                        "internal", "")]

    plan = 'hits = search("paybridge degradation", top_k=3)\nresult = [h["chunk_id"] for h in hits]'
    r = await run_plan(plan, Catalog(VIEWER, retriever=FakeRetriever()))
    assert r.ok, r.error
    assert r.result == ["RB-003#actions"] and r.catalog_usage["search"] == 1


# ---------- memory: single-step blow-ups are refused before they allocate ----------

@pytest.mark.parametrize("code", [
    'result = len("a" * 100000 * 100000)',                     # str * int
    "result = len([0] * 100000 * 100000)",                     # list * int
    'x = "a" * 1000\nx *= 100000\nresult = len(x)',           # *=
    'x = "a" * 600000\nresult = len(x + x)',                  # str + str
    'x = ["a" * 100000] * 50\nresult = len("".join(x))',      # join of repeated references
    'x = "a" * 100000\nresult = len(x.replace("a", "b" * 1000))',  # replace amplification
    "result = sum([[0] * 1000] * 2000, [])",                   # sum of lists
    'result = len("x".ljust(100000 * 100000))',               # padding method
    'result = f"{1:>100000000}"',                              # wide format spec
    'w = 9\nresult = f"{1:{w}}"',                              # computed format spec
    'result = "%100000000d" % 1',                              # printf width (literal)
    'p = "%100000000d"\nresult = p % 1',                       # printf width (variable)
    "result = 100000 * 100000 * 100000 * 100000",              # huge integers
])
async def test_memory_blowups_are_refused(catalog, code):
    r = await run_plan(code, catalog)
    assert not r.ok, code
    assert any(w in r.error for w in ("max 1,000,000", "not allowed", "numbers only", "too large", "format")), r.error


async def test_gradual_memory_growth_hits_the_memory_limit(catalog):
    """list.extend doubles a list line by line – each step is small, so the per-process memory limit stops it."""
    plan = "x = [0] * 100000\nfor i in range(40):\n    x.extend(x)\nresult = len(x)"
    r = await run_plan(plan, catalog, max_memory_mb=64)
    assert not r.ok and "memory limit" in r.error


async def test_normal_string_and_list_work_still_runs(catalog):
    plan = ('ids = [d["doc_id"] for d in list_docs(document_type="incident")]\n'
            'label = ", ".join(ids[:3])\n'
            'total = sum([1, 2, 3.5])\n'
            'title = "Report: " + label.replace(",", ";")\n'
            'counts = {}\nfor i in ids:\n    counts[i[:3]] = counts.get(i[:3], 0) + 1\n'
            'result = {"title": title, "total": total, "pct": f"{total / 3:.1f}", "n": counts}')
    r = await run_plan(plan, catalog)
    assert r.ok, r.error
    assert r.result["title"].startswith("Report: INC-") and r.result["pct"] == "2.2" and r.result["n"]["INC"] > 5


async def test_bare_underscore_is_allowed_as_a_throwaway_name(catalog):
    r = await run_plan("result = [0 for _ in range(3)]", catalog)
    assert r.ok and r.result == [0, 0, 0]
    r = await run_plan("__x = 1\nresult = __x", catalog)
    assert not r.ok and "'_'" in r.error
