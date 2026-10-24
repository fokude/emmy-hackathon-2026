import sys
import json
from pathlib import Path
from emmy.compiler.graph import Graph

VISIBLE = ".visible"
ENTRY = ".entry"
PARAM_DECLARATION = ".param"
PARAM_PREFIX = PARAM_DECLARATION[1:] + "_"

U64_TYPE = ".u64"
TAB_SPACE = 2
SPACE_CHAR = ' '

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

def parse_graph(graph, metadata):
    for kernel in metadata["kernels"]:
        parse_kernel(kernel)

def parse_kernel(kernel):
    entry_name = kernel["entry"]
    arg_names = kernel["arg_order"]
    
    print(f"""
{VISIBLE} {ENTRY} {entry_name}(
{parse_params(arg_names, kernel)}
)
    """)

def parse_params(arg_names, kernel):
    # ask if metadata is guaranteed to always have same num args as arg order, and if args in args will be in same order as arg order..
    arg_dict = {}
    for arg in kernel["args"]:
        arg_dict[arg["name"]] = arg
    
    param_strings = []
    for idx, arg_name in enumerate(arg_names):
        arg = arg_dict[arg_name]
        param_string = f"{SPACE_CHAR * TAB_SPACE} {PARAM_DECLARATION} {U64_TYPE} {PARAM_PREFIX + arg["name"]}"
        param_strings.append(param_string)
    return ', \n'.join(param_strings)
    
# IR and Metadata successfully loaded, proceed with compilation.
# Print constants
print(".version 8.0")
print(".target sm_89")
print(".address_size 64")

problem_graph = Graph.from_dict(ir_data)
parse_graph(problem_graph, metadata)

"""
tile = problem_graph.nodes["add"].op.body[0]
for stmt in tile.body:
    continue
    print(type(stmt).__name__)
"""
