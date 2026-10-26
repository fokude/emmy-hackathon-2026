"""Translate each IR statement of a tile body into PTX lines."""

from emmy.compiler.ir.stmt import StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Write
from ptxgen.address import emit_element_address, emit_expr
from ptxgen.consts import (
    ADD,
    BRANCH,
    ELEMENTWISE_OPS,
    GLOBAL,
    GREATER_EQUALS,
    INDENT,
    LOAD,
    LOOP_END_LABEL,
    LOOP_LABEL,
    MOVE,
    SET_PREDICATE,
    STORE,
)
from ptxgen.labels import LabelAllocator
from ptxgen.regs import RegisterAllocator
from ptxgen.types import DTYPE_INFO, PtxType, f32_literal


def emit_load(load, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the lines that load one tensor element into a new register."""
    assert len(load.names) == 1, "ASSUMPTIONS 11: a Load defines exactly one value"
    assert len(load.index) == 1, "ASSUMPTIONS 10: a Load index has exactly one expression"
    ptx_type, reg_kind = DTYPE_INFO[load.dtype.name]
    address_reg, lines = emit_element_address(load.input, load.index[0], load.dtype.nbytes, reg_alloc)
    value_reg = reg_alloc.new_reg(reg_kind)
    lines.append(f"{INDENT}{LOAD}{GLOBAL}{ptx_type} {value_reg}, [{address_reg}];")
    reg_alloc.bind(load.names[0], value_reg)
    return lines


def emit_assign(assign, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the line that applies an elementwise op to registers into a new register."""
    ptx_type, reg_kind = DTYPE_INFO[assign.dtype.name]
    operands = [reg_alloc.lookup(arg) for arg in assign.args]
    result_reg = reg_alloc.new_reg(reg_kind)
    instruction, extra_operands = ELEMENTWISE_OPS[assign.op.name]
    line = f"{INDENT}{instruction}{ptx_type} {result_reg}, {', '.join(operands + extra_operands)};"
    reg_alloc.bind(assign.name, result_reg)
    return [line]


def emit_write(write, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the lines that store the computed values to the output tensor."""
    assert len(write.index) == 1, "ASSUMPTIONS 10: a Write index has exactly one expression"
    assert len(write.values) in (2, 4), "ASSUMPTIONS 12: a Write stores 2 or 4 values (vector store)"
    ptx_type, _ = DTYPE_INFO[write.value_dtype.name]
    address_reg, lines = emit_element_address(write.output, write.index[0], write.value_dtype.nbytes, reg_alloc)
    value_regs = [reg_alloc.lookup(name) for name in write.values]
    lines.append(
        f"{INDENT}{STORE}{GLOBAL}.v{len(value_regs)}{ptx_type} [{address_reg}], {{{', '.join(value_regs)}}};"
    )
    return lines


def emit_accum_seeds(loop, reg_alloc: RegisterAllocator) -> list[str]:
    """Return one line per Accum in the loop body that sets its register to the op's identity."""
    assert loop.seed, "ASSUMPTIONS 16: a StridedLoop seeds its accumulators (seed is True)"
    lines = []
    for stmt in loop.body:
        if isinstance(stmt, Accum):
            ptx_type, reg_kind = DTYPE_INFO[stmt.dtype.name]
            acc_reg = reg_alloc.new_reg(reg_kind)
            lines.append(f"{INDENT}{MOVE}{ptx_type} {acc_reg}, {f32_literal(stmt.op.identity)};")
            reg_alloc.bind(stmt.name, acc_reg)
    return lines


def emit_accum(accum, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the instruction for acc = op(acc, value), storing the result back in the accumulator's own register."""
    ptx_type, _ = DTYPE_INFO[accum.dtype.name]
    acc_reg = reg_alloc.lookup(accum.name)
    value_reg = reg_alloc.lookup(accum.value)
    instruction, _ = ELEMENTWISE_OPS[accum.op.name]
    return [f"{INDENT}{instruction}{ptx_type} {acc_reg}, {acc_reg}, {value_reg};"]


def emit_strided_loop(loop, reg_alloc: RegisterAllocator, labels: LabelAllocator) -> list[str]:
    """Return the lines for a loop whose variable starts at loop.start and grows by loop.step."""
    assert loop.end is None, "ASSUMPTIONS 15: a StridedLoop stops at its axis extent (end is None)"
    start_operand, start_lines = emit_expr(loop.start, reg_alloc)
    step_operand, step_lines = emit_expr(loop.step, reg_alloc)
    extent = loop.axis.extent.as_static()
    counter_reg = reg_alloc.new_reg('r')
    exit_predicate_reg = reg_alloc.new_reg('p')
    loop_label = labels.new_label(LOOP_LABEL)
    end_label = labels.new_label(LOOP_END_LABEL)
    reg_alloc.bind(loop.axis.name, counter_reg)
    return [
        *start_lines,
        *step_lines,
        f"{INDENT}{MOVE}{PtxType.U32} {counter_reg}, {start_operand};",
        *emit_accum_seeds(loop, reg_alloc),
        f"{loop_label}:",
        f"{INDENT}{SET_PREDICATE}.{GREATER_EQUALS}{PtxType.U32} {exit_predicate_reg}, {counter_reg}, {extent};",
        f"{INDENT}@{exit_predicate_reg} {BRANCH} {end_label};",
        *emit_statements(loop.body, reg_alloc, labels),
        f"{INDENT}{ADD}{PtxType.U32} {counter_reg}, {counter_reg}, {step_operand};",
        f"{INDENT}{BRANCH} {loop_label};",
        f"{end_label}:",
    ]


def emit_statements(body, reg_alloc: RegisterAllocator, labels: LabelAllocator) -> list[str]:
    """Return the PTX for every statement in a body, in order."""
    lines = []
    for stmt in body:
        match stmt:
            case Load():
                lines += emit_load(stmt, reg_alloc)
            case Assign():
                lines += emit_assign(stmt, reg_alloc)
            case Write():
                lines += emit_write(stmt, reg_alloc)
            case Accum():
                lines += emit_accum(stmt, reg_alloc)
            case StridedLoop():
                lines += emit_strided_loop(stmt, reg_alloc, labels)
            case _:
                raise NotImplementedError(f"statement {type(stmt).__name__}")
    return lines
