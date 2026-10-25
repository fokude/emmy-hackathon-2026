import sys
import json
from pathlib import Path
from emmy.compiler.graph import Graph
from ptxgen.emit import emit_graph

if len(sys.argv) < 2:
    print("usage: ./venv/bin/python compile.py ./problems/<problem>/<problem>.ir.json")
    sys.exit(0)

ir_path = Path(sys.argv[1]).resolve()
meta_path = ir_path.with_name("meta.json")

if not ir_path.exists():
    sys.exit(f"[err]: file does not exist: {ir_path}")
if not ir_path.is_file():
    sys.exit(f"[err]: target path is not a file: {ir_path}")

try:
    with open(ir_path, "r", encoding="utf-8") as f:
        ir_data = json.load(f)
except json.JSONDecodeError as err:
        sys.exit(f"[err]: malformed JSON in {ir_path.name}: {err}")

try:
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
except json.JSONDecodeError as err:
        sys.exit(f"[err]: malformed JSON in {meta_path.name}: {err}")

# IR and Metadata successfully loaded, proceed with compilation.
# Print constants
print(".version 8.0")
print(".target sm_89")
print(".address_size 64")

problem_graph = Graph.from_dict(ir_data)
emit_graph(problem_graph, metadata)

"""
tile = problem_graph.nodes["add"].op.body[0]
for stmt in tile.body:
    continue
    print(type(stmt).__name__)
"""
