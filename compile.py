import sys
import json
from pathlib import Path
from emmy.compiler.graph import Graph

if len(sys.argv) < 2:
    print("usage: ./venv/bin/python compile.py ./problems/<problem>/<problem>.ir.json")
    sys.exit(0)

ir_path = Path(sys.argv[1]).resolve()

if not ir_path.exists():
    sys.exit(f"[err]: file does not exist: {ir_path}")
if not ir_path.is_file():
    sys.exit(f"[err]: target path is not a file: {ir_path}")

try:
    with open(ir_path, "r", encoding="utf-8") as f:
        ir_data = json.load(f)
except json.JSONDecodeError as err:
        sys.exit(f"[err]: malformed JSON in {ir_path.name}: {err}")

problem_graph = Graph.from_dict(ir_data)
tile = problem_graph.nodes["add"].op.body[0]
for stmt in tile.body:
    print(type(stmt).__name__)
