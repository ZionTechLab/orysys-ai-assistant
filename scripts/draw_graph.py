"""Print the agent graph as Mermaid (diagram D2), generated from the compiled graph.

Run: .venv/Scripts/python scripts/draw_graph.py
Paste the output into docs/agents.md so the diagram always matches the code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph.builder import build_graph  # noqa: E402

print(build_graph().get_graph().draw_mermaid())
