"""Sync data/docs into Pinecone (hybrid: dense + BM25 sparse), one namespace per department.

Run:
  .venv/Scripts/python scripts/ingest.py             # sync: only new / changed / deleted documents
  .venv/Scripts/python scripts/ingest.py --dry-run   # show what would change, change nothing
  .venv/Scripts/python scripts/ingest.py --full      # re-embed every document

See app/retrieval/sync.py and decision A11.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.logging import configure_logging  # noqa: E402
from app.retrieval.sync import sync_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    parser.add_argument("--full", action="store_true", help="re-embed all documents")
    args = parser.parse_args()

    s = get_settings()
    configure_logging(s.log_level, "console")
    report = asyncio.run(sync_index(dry_run=args.dry_run, force=args.full))

    r = report.summary()
    print("\nSync" + (" (dry run)" if report.dry_run else "") + f" finished in {r['seconds']}s")
    print(f"  added     : {r['added'] or '-'}")
    print(f"  updated   : {r['updated'] or '-'}")
    print(f"  deleted   : {r['deleted'] or '-'}")
    print(f"  unchanged : {r['unchanged']} documents")
    print(f"  chunks    : {r['chunks_upserted']} upserted, {r['chunks_deleted']} deleted, "
          f"{r['sparse_refreshed']} sparse refreshed, BM25 re-fit: {r['bm25_refit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
