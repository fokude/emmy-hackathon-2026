from emmy.compiler.graph import Graph
from ptxgen.consts import (
    VISIBLE,
    ENTRY,
    PARAM_DECLARATION,
    PARAM_PREFIX,
    U64_TYPE,
    TAB_SPACE,
    SPACE_CHAR,
    REG_DECL_TYPES,
    U32_TYPE,
    MOVE,
    DONE_LABEL,
    SET_PREDICATE,
    GREATER_EQUALS,
    BRANCH,
    MULTIPLY_ADD,
    LOWER_BITS,
)
from ptxgen.regs import RegisterAllocator

def emit_graph(graph: Graph, metadata: dict) -> list[str]:
    lines = []
    for kernel in metadata["kernels"]:
        lines += emit_kernel(graph, kernel)
    return lines

def emit_kernel(graph: Graph, kernel: dict) -> list[str]:
    reg_alloc = RegisterAllocator()
    # run emit_kernel body instead of inline with returns so register allocator can be populated before emiting register declarations
    body = emit_kernel_body(graph, kernel, reg_alloc)
    return [
        *emit_kernel_signature(kernel),
        "{",
        *emit_reg_decls(reg_alloc),
        *body,
        "}",
    ]

def emit_kernel_signature(kernel: dict) -> list[str]:
    entry_name = kernel["entry"]
    arg_names = kernel["arg_order"]

    return [
        "",
        f"{VISIBLE} {ENTRY} {entry_name}(",
        *emit_params(arg_names, kernel),
        ")",
    ]

def emit_kernel_body(graph: Graph, kernel: dict, reg_alloc: RegisterAllocator) -> list[str]:
    tile = graph.nodes[kernel["node"]].op.body[0]
    return [
        *emit_prologue(tile, reg_alloc),
        f"{DONE_LABEL}:",
        f"{SPACE_CHAR * TAB_SPACE}ret;",
    ]

def emit_reg_decls(reg_alloc: RegisterAllocator) -> list[str]:
    return [
        f"{SPACE_CHAR * TAB_SPACE}.reg {REG_DECL_TYPES[kind]} %{kind}<{count}>;" for kind, count in reg_alloc.counts().items()
    ]

def emit_params(arg_names: list[str], kernel: dict) -> list[str]:
    # ask if metadata is guaranteed to always have same num args as arg order, and if args in args will be in same order as arg order..
    arg_dict = {}
    for arg in kernel["args"]:
        arg_dict[arg["name"]] = arg

    param_lines = []
    for idx, arg_name in enumerate(arg_names):
        arg = arg_dict[arg_name]
        comma = "," if idx < len(arg_names) - 1 else ""
        param_line = f"{SPACE_CHAR * TAB_SPACE} {PARAM_DECLARATION} {U64_TYPE} {PARAM_PREFIX + arg["name"]}{comma}"
        param_lines.append(param_line)
    return param_lines


def emit_prologue(tile, reg_alloc):
    assert(len(tile.axes) == 1)

    indent = SPACE_CHAR * TAB_SPACE
    block_id_reg = reg_alloc.new_reg('r')
    num_threads_reg = reg_alloc.new_reg('r')
    thread_id_reg = reg_alloc.new_reg('r')
    gid_reg = reg_alloc.new_reg('r')
    thread_boundary_reg = reg_alloc.new_reg('p')

    return [
        f"{indent}{MOVE}{U32_TYPE} {block_id_reg}, %ctaid.x;",
        f"{indent}{MOVE}{U32_TYPE} {num_threads_reg}, %ntid.x;",
        f"{indent}{MOVE}{U32_TYPE} {thread_id_reg}, %tid.x;",
        f"{indent}{MULTIPLY_ADD}.{LOWER_BITS}{U32_TYPE} {gid_reg}, {block_id_reg}, {num_threads_reg}, {thread_id_reg};",
        f"{indent}{SET_PREDICATE}.{GREATER_EQUALS}{U32_TYPE} {thread_boundary_reg}, {gid_reg}, {tile.axes[0].extent.as_static()};",
        f"{indent}@{thread_boundary_reg} {BRANCH} {DONE_LABEL};",
    ]
