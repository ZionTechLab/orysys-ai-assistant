"""BM25 sparse encoder (keyword search side of hybrid retrieval).

Fitted on our own corpus so term weights reflect bank vocabulary
(e.g. "PayBridge", "connection pool"). Saved to disk so ingestion and queries
use exactly the same vocabulary statistics.
"""

from functools import lru_cache
from pathlib import Path

from pinecone_text.sparse import BM25Encoder

from app.config import get_settings


def fit_bm25(texts: list[str], path: str | Path) -> BM25Encoder:
    encoder = BM25Encoder()
    encoder.fit(texts)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    encoder.dump(str(path))
    return encoder


@lru_cache
def get_bm25() -> BM25Encoder:
    path = Path(get_settings().bm25_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found – run scripts/ingest.py first")
    return BM25Encoder().load(str(path))
