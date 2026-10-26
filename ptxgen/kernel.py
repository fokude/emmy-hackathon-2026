"""Emit the PTX for one kernel: signature, registers, prologue, parameter loads and body."""

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import Tile
from ptxgen.consts import (
    VISIBLE,
    ENTRY,
    PARAM,
    PARAM_PREFIX,
    REG_DECL_TYPES,
    MOVE,
    LOAD,
    GLOBAL,
    CONVERT_ADDRESS,
    INDENT,
    DONE_LABEL,
    SET_PREDICATE,
    GREATER_EQUALS,
    BRANCH,
    MULTIPLY_ADD,
    LOWER_BITS,
)
from ptxgen.labels import LabelAllocator
from ptxgen.regs import RegisterAllocator
from ptxgen.statements import emit_statements
from ptxgen.types import PtxType

def emit_kernel(graph: Graph, kernel: dict) -> list[str]:
    """Return the PTX lines for one kernel.

    The body is built before the .reg lines, because the register counts are
    only known once every instruction has been emitted.
    """
    assert kernel["grid"][1:] == [1, 1] and kernel["block"][1:] == [1, 1], "ASSUMPTIONS 7: 1-D launch only"
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
    body = graph.nodes[kernel["node"]].op.body
    assert len(body) == 1 and isinstance(body[0], Tile), "ASSUMPTIONS 4: kernel body must be a single Tile"
    tile = body[0]
    # block_threads is None for a scalar tile (one thread per cell, axis = global thread id).
    # A number means a cooperative tile: that many threads work together on one result.
    if tile.block_threads is None:
        prologue = emit_prologue(tile, reg_alloc)
    else:
        prologue = emit_cooperative_prologue(tile, kernel, reg_alloc)
    labels = LabelAllocator()
    return [
        *prologue,
        *emit_param_loads(kernel, reg_alloc),
        *emit_statements(tile.body, reg_alloc, labels),
        f"{DONE_LABEL}:",
        f"{INDENT}ret;",
    ]

def emit_reg_decls(reg_alloc: RegisterAllocator) -> list[str]:
    """Return one .reg line per register kind the allocator handed out."""
    return [
        f"{INDENT}.reg {REG_DECL_TYPES[kind]} %{kind}<{count}>;"
        for kind, count in reg_alloc.counts().items()
    ]

def emit_params(arg_names: list[str], kernel: dict) -> list[str]:
    """Return one .param line per kernel argument, in arg_order."""
    # ask if metadata is guaranteed to always have same num args as arg order, and if args in args will be in same order as arg order..
    arg_dict = {}
    for arg in kernel["args"]:
        arg_dict[arg["name"]] = arg
    assert set(arg_dict) == set(arg_names), "ASSUMPTIONS 2: args and arg_order must list the same names"

    param_lines = []
    for idx, arg_name in enumerate(arg_names):
        arg = arg_dict[arg_name]
        comma = "," if idx < len(arg_names) - 1 else ""
        param_line = f"{INDENT}{PARAM} {PtxType.U64} {PARAM_PREFIX}{arg_name}{comma}"
        param_lines.append(param_line)
    return param_lines


def emit_param_loads(kernel: dict, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the lines that load each argument's address into a global-space register."""
    lines = []
    for arg_name in kernel["arg_order"]:
        param_reg = reg_alloc.new_reg('rd')
        global_reg = reg_alloc.new_reg('rd')
        lines.append(f"{INDENT}{LOAD}{PARAM}{PtxType.U64} {param_reg}, [{PARAM_PREFIX}{arg_name}];")
        lines.append(f"{INDENT}{CONVERT_ADDRESS}.to{GLOBAL}{PtxType.U64} {global_reg}, {param_reg};")
        reg_alloc.bind_tensor(arg_name, global_reg)
    return lines


def emit_prologue(tile, reg_alloc):
    """Return the lines that compute the global thread id and guard the tile.

    Threads whose id is at or past the tile extent branch to DONE.
    """
    assert len(tile.axes) == 1, "ASSUMPTIONS 5: a Tile has exactly one axis"

    block_id_reg = reg_alloc.new_reg('r')
    num_threads_reg = reg_alloc.new_reg('r')
    thread_id_reg = reg_alloc.new_reg('r')
    gid_reg = reg_alloc.new_reg('r')
    out_of_range_predicate_reg = reg_alloc.new_reg('p')

    axis = tile.axes[0]
    extent = axis.extent.as_static()
    # Emmy's Tile runs one thread per axis value, so the axis variable (a0)
    # is the global thread id. Record that for the body's index expressions.
    reg_alloc.bind(axis.name, gid_reg)

    return [
        f"{INDENT}{MOVE}{PtxType.U32} {block_id_reg}, %ctaid.x;",
        f"{INDENT}{MOVE}{PtxType.U32} {num_threads_reg}, %ntid.x;",
        f"{INDENT}{MOVE}{PtxType.U32} {thread_id_reg}, %tid.x;",
        f"{INDENT}{MULTIPLY_ADD}.{LOWER_BITS}{PtxType.U32} {gid_reg}, {block_id_reg}, {num_threads_reg}, {thread_id_reg};",
        f"{INDENT}{SET_PREDICATE}.{GREATER_EQUALS}{PtxType.U32} {out_of_range_predicate_reg}, {gid_reg}, {extent};",
        f"{INDENT}@{out_of_range_predicate_reg} {BRANCH} {DONE_LABEL};",
    ]


def emit_cooperative_prologue(tile, kernel, reg_alloc):
    """Return the line that reads the thread id for a cooperative tile.

    The launch is one block whose size equals the axis extent, so the axis
    variable (a0_co) is tid.x and no bounds guard is needed.
    """
    assert len(tile.axes) == 1, "ASSUMPTIONS 5: a Tile has exactly one axis"
    axis = tile.axes[0]
    assert kernel["grid"] == [1, 1, 1], "ASSUMPTIONS 14: a cooperative tile launches exactly one block"
    assert kernel["block"][0] == axis.extent.as_static(), "ASSUMPTIONS 14: block size equals the cooperative axis extent"

    thread_id_reg = reg_alloc.new_reg('r')
    reg_alloc.bind(axis.name, thread_id_reg)

    return [f"{INDENT}{MOVE}{PtxType.U32} {thread_id_reg}, %tid.x;"]
