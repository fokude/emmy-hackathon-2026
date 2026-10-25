"""Walk the Kernel IR and emit PTX as lists of lines."""

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from ptxgen.consts import (
    VISIBLE,
    ENTRY,
    PARAM,
    PARAM_PREFIX,
    TAB_SPACE,
    SPACE_CHAR,
    REG_DECL_TYPES,
    MOVE,
    LOAD,
    CONVERT_ADDRESS,
    INDENT,
    BINARY_OPS,
    MULTIPLY_WIDE,
    ADD,
    DONE_LABEL,
    SET_PREDICATE,
    GREATER_EQUALS,
    BRANCH,
    MULTIPLY_ADD,
    LOWER_BITS,
)
from ptxgen.regs import RegisterAllocator
from ptxgen.types import PtxType

def emit_graph(graph: Graph, metadata: dict) -> list[str]:
    """Return the PTX lines for every kernel listed in the metadata."""
    lines = []
    for kernel in metadata["kernels"]:
        lines += emit_kernel(graph, kernel)
    return lines

def emit_kernel(graph: Graph, kernel: dict) -> list[str]:
    """Return the PTX lines for one kernel.

    The body is built before the .reg lines, because the register counts are
    only known once every instruction has been emitted.
    """
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
    """Return the .visible .entry line and the parameter list."""
    entry_name = kernel["entry"]
    arg_names = kernel["arg_order"]

    return [
        "",
        f"{VISIBLE} {ENTRY} {entry_name}(",
        *emit_params(arg_names, kernel),
        ")",
    ]

def emit_kernel_body(graph: Graph, kernel: dict, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the kernel instructions, ending with the DONE label and ret."""
    tile = graph.nodes[kernel["node"]].op.body[0]
    return [
        *emit_prologue(tile, reg_alloc),
        *emit_param_loads(kernel, reg_alloc),
        f"{DONE_LABEL}:",
        f"{SPACE_CHAR * TAB_SPACE}ret;",
    ]

def emit_reg_decls(reg_alloc: RegisterAllocator) -> list[str]:
    """Return one .reg line per register kind the allocator handed out."""
    return [
        f"{SPACE_CHAR * TAB_SPACE}.reg {REG_DECL_TYPES[kind]} %{kind}<{count}>;" for kind, count in reg_alloc.counts().items()
    ]

def emit_params(arg_names: list[str], kernel: dict) -> list[str]:
    """Return one .param line per kernel argument, in arg_order."""
    # ask if metadata is guaranteed to always have same num args as arg order, and if args in args will be in same order as arg order..
    arg_dict = {}
    for arg in kernel["args"]:
        arg_dict[arg["name"]] = arg

    param_lines = []
    for idx, arg_name in enumerate(arg_names):
        arg = arg_dict[arg_name]
        comma = "," if idx < len(arg_names) - 1 else ""
        param_line = f"{SPACE_CHAR * TAB_SPACE} {PARAM} {PtxType.U64} {PARAM_PREFIX + arg["name"]}{comma}"
        param_lines.append(param_line)
    return param_lines


def emit_param_loads(kernel: dict, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the lines that load each argument's address into a global-space register."""
    lines = []
    for arg_name in kernel["arg_order"]:
        param_reg = reg_alloc.new_reg('rd')
        global_reg = reg_alloc.new_reg('rd')
        lines.append(f"{INDENT}{LOAD}{PARAM}{PtxType.U64} {param_reg}, [{PARAM_PREFIX}{arg_name}];")
        lines.append(f"{INDENT}{CONVERT_ADDRESS}.to.global{PtxType.U64} {global_reg}, {param_reg};")
        reg_alloc.bind_tensor(arg_name, global_reg)
    return lines


def emit_expr(expr, reg_alloc: RegisterAllocator) -> tuple[str, list[str]]:
    """Return (operand, lines) that evaluate an IR index expression."""
    match expr:
        case Var(name=name):
            return reg_alloc.lookup(name), []
        case Literal(value=value):
            return str(value), []
        case BinaryExpr(op=op, left=left, right=right):
            left_operand, left_lines = emit_expr(left, reg_alloc)
            right_operand, right_lines = emit_expr(right, reg_alloc)
            result_reg = reg_alloc.new_reg('r')
            instruction = BINARY_OPS[op]
            line = f"{INDENT}{instruction}{PtxType.U32} {result_reg}, {left_operand}, {right_operand};"
            # Post-order: the operands' lines come before the instruction that reads them.
            return result_reg, [*left_lines, *right_lines, line]
        case _:
            raise NotImplementedError(f"expression {type(expr).__name__}")


def emit_address(tensor_name, index_expr, bytes_per_element, reg_alloc):
    """Return (address_reg, lines) for element index_expr of a tensor."""
    index_operand, lines = emit_expr(index_expr, reg_alloc)
    offset_reg = reg_alloc.new_reg('rd')
    address_reg = reg_alloc.new_reg('rd')
    base_reg = reg_alloc.lookup_tensor(tensor_name)
    lines += [
        f"{INDENT}{MULTIPLY_WIDE}{PtxType.U32} {offset_reg}, {index_operand}, {bytes_per_element};",
        f"{INDENT}{ADD}{PtxType.U64} {address_reg}, {base_reg}, {offset_reg};",
    ]
    return address_reg, lines


def emit_prologue(tile, reg_alloc):
    """Return the lines that compute the global thread id and guard the tile.

    Threads whose id is at or past the tile extent branch to DONE.
    """
    assert(len(tile.axes) == 1)

    indent = SPACE_CHAR * TAB_SPACE
    block_id_reg = reg_alloc.new_reg('r')
    num_threads_reg = reg_alloc.new_reg('r')
    thread_id_reg = reg_alloc.new_reg('r')
    gid_reg = reg_alloc.new_reg('r')
    thread_boundary_reg = reg_alloc.new_reg('p')

    axis = tile.axes[0]
    # Emmy's Tile runs one thread per axis value, so the axis variable (a0)
    # is the global thread id. Record that for the body's index expressions.
    reg_alloc.bind(axis.name, gid_reg)

    return [
        f"{indent}{MOVE}{PtxType.U32} {block_id_reg}, %ctaid.x;",
        f"{indent}{MOVE}{PtxType.U32} {num_threads_reg}, %ntid.x;",
        f"{indent}{MOVE}{PtxType.U32} {thread_id_reg}, %tid.x;",
        f"{indent}{MULTIPLY_ADD}.{LOWER_BITS}{PtxType.U32} {gid_reg}, {block_id_reg}, {num_threads_reg}, {thread_id_reg};",
        f"{indent}{SET_PREDICATE}.{GREATER_EQUALS}{PtxType.U32} {thread_boundary_reg}, {gid_reg}, {tile.axes[0].extent.as_static()};",
        f"{indent}@{thread_boundary_reg} {BRANCH} {DONE_LABEL};",
    ]
