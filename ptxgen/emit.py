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
    body = emit_kernel_body(graph, reg_alloc)
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

def emit_kernel_body(graph: Graph, regs: RegisterAllocator) -> list[str]:
    return []

def emit_reg_decls(regs: RegisterAllocator) -> list[str]:
    return [
        f"{SPACE_CHAR * TAB_SPACE}.reg {REG_DECL_TYPES[kind]} %{kind}<{count}>;"
        for kind, count in regs.counts().items()
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
