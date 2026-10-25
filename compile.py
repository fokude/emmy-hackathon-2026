"""Command-line entry point: compile a Kernel IR json file to PTX on stdout."""

import sys
import json
from pathlib import Path
from emmy.compiler.graph import Graph
from ptxgen.emit import emit_graph

def main():
    """Print the PTX header, then the PTX for every kernel in the problem."""
    ir_data, metadata = parse_args()
    print_header()
    problem_graph = Graph.from_dict(ir_data)
    print("\n".join(emit_graph(problem_graph, metadata)))

def parse_args():
    """Read the IR path from argv and return the parsed (ir_data, metadata).

    Exits with a message on a missing, unreadable or malformed file.
    """
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

    return (ir_data, metadata)

def print_header():
    """Print the PTX version, target and address size lines."""
    print(".version 8.0")
    print(".target sm_89")
    print(".address_size 64")

if __name__ == "__main__":
    main()

"""
tile = problem_graph.nodes["add"].op.body[0]
for stmt in tile.body:
    continue
    print(type(stmt).__name__)
"""
