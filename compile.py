import sys
import json
from emmy.compiler.graph import Graph

if len(sys.argv) <= 1:
    print("Missing path")
    sys.exit(0)

file_path = sys.argv[1]
with open(file_path, "r", encoding="utf-8") as f:
    ir_data = json.load(f)

graph = Graph.from_dict(ir_data)

#for node_id, node in graph.nodes.items():
#    print(node_id, node, "\n")


tile = graph.nodes["add"].op.body[0]
for stmt in tile.body:
    print(type(stmt).__name__)
