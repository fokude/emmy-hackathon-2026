"""Translate each IR statement of a tile body into PTX lines."""

from emmy.compiler.ir.expr import BinaryExpr, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Write
from ptxgen.address import emit_element_address, emit_expr, emit_shared_address
from ptxgen.consts import (
    ADD,
    ALIGN,
    BARRIER_SYNC,
    BRANCH,
    COND_END_LABEL,
    ELEMENTWISE_OPS,
    F32_BYTES,
    FULL_WARP_MASK,
    GLOBAL,
    GREATER_EQUALS,
    INDENT,
    LOAD,
    LOOP_END_LABEL,
    LOOP_LABEL,
    MOVE,
    NOT_EQUALS,
    SET_PREDICATE,
    SHARED,
    SHUFFLE_BUTTERFLY,
    SHUFFLE_MAX_LANE,
    STORE,
    TREE_END_LABEL,
    WARP_SIZE,
)
from ptxgen.labels import LabelAllocator
from ptxgen.regs import RegisterAllocator
from ptxgen.types import DTYPE_INFO, PtxType, f32_literal


def emit_load(load, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the lines that load one tensor element into a new register."""
    assert len(load.names) == 1, "ASSUMPTIONS 11: a Load defines exactly one value"
    ptx_type, reg_kind = DTYPE_INFO[load.dtype.name]
    address_reg, lines = emit_element_address(load.input, load.index, load.dtype.nbytes, reg_alloc)
    value_reg = reg_alloc.new_reg(reg_kind)
    lines.append(f"{INDENT}{LOAD}{GLOBAL}{ptx_type} {value_reg}, [{address_reg}];")
    reg_alloc.bind(load.names[0], value_reg)
    return lines


def format_op_line(op_name: str, ptx_type: str, dest_reg: str, operand_regs: list[str]) -> str:
    """Return the instruction line that applies an elementwise op to operand_regs and writes dest_reg."""
    instruction, extra_operands = ELEMENTWISE_OPS[op_name]
    return f"{INDENT}{instruction}{ptx_type} {dest_reg}, {', '.join(operand_regs + extra_operands)};"


def emit_assign(assign, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the line that applies an elementwise op to registers into a new register."""
    ptx_type, reg_kind = DTYPE_INFO[assign.dtype.name]
    operands = [reg_alloc.lookup(arg) for arg in assign.args]
    result_reg = reg_alloc.new_reg(reg_kind)
    reg_alloc.bind(assign.name, result_reg)
    return [format_op_line(assign.op.name, ptx_type, result_reg, operands)]


def emit_write(write, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the lines that store computed values to a shared array or an output tensor."""
    if reg_alloc.is_shared(write.output):
        return emit_shared_write(write, reg_alloc)
    return emit_global_write(write, reg_alloc)


def emit_global_write(write, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the lines that store 1, 2 or 4 values to an output tensor in global memory."""
    assert len(write.values) in (1, 2, 4), "ASSUMPTIONS 12: a global Write stores 1, 2 or 4 values"
    ptx_type, _ = DTYPE_INFO[write.value_dtype.name]
    if write.index:
        address_reg, lines = emit_element_address(write.output, write.index, write.value_dtype.nbytes, reg_alloc)
    else:
        address_reg, lines = reg_alloc.lookup_tensor(write.output), []
    value_regs = [reg_alloc.lookup(name) for name in write.values]
    if len(value_regs) == 1:
        vector_suffix, store_operand = "", value_regs[0]
    else:
        vector_suffix, store_operand = f".v{len(value_regs)}", f"{{{', '.join(value_regs)}}}"
    lines.append(f"{INDENT}{STORE}{GLOBAL}{vector_suffix}{ptx_type} [{address_reg}], {store_operand};")
    return lines


def emit_shared_write(write, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the lines that store one value into a slot of a shared array."""
    assert len(write.index) == 1, "ASSUMPTIONS 10: a Write index has exactly one expression"
    assert len(write.values) == 1, "ASSUMPTIONS 20: a shared Write stores one value"
    ptx_type, _ = DTYPE_INFO[write.value_dtype.name]
    address_reg, lines = emit_shared_address(write.output, write.index[0], write.value_dtype.nbytes, reg_alloc)
    lines.append(f"{INDENT}{STORE}{SHARED}{ptx_type} [{address_reg}], {reg_alloc.lookup(write.values[0])};")
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
    return [format_op_line(accum.op.name, ptx_type, acc_reg, [acc_reg, value_reg])]


def emit_warp_shuffle(shuffle, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the lines that combine state across the 32 lanes of a warp, halving the lane mask from 16 to 1."""
    assert shuffle.length == WARP_SIZE, "ASSUMPTIONS 17: a WarpShuffle spans exactly one warp (32 lanes)"
    assert len(shuffle.state) == 1 and len(shuffle.combine_states) == 1, "ASSUMPTIONS 17: a WarpShuffle has one state value"
    ptx_type, reg_kind = DTYPE_INFO[shuffle.dtype.name]
    acc_reg = reg_alloc.lookup(shuffle.state[0])
    lane_partner_acc_reg = reg_alloc.new_reg(reg_kind)
    reg_alloc.bind(shuffle.state_b[0], lane_partner_acc_reg)
    combine = shuffle.combine_states[0]
    operands = [reg_alloc.lookup(arg) for arg in combine.args]

    lines = []
    # lane_mask 16, 8, 4, 2, 1: each step flips one bit of the lane number, so after 5 steps every lane has seen all 32
    lane_mask = shuffle.length // 2
    while lane_mask >= 1:
        lines += [
            f"{INDENT}{SHUFFLE_BUTTERFLY}{PtxType.B32} {lane_partner_acc_reg}, {acc_reg}, {lane_mask}, {SHUFFLE_MAX_LANE}, {FULL_WARP_MASK};",
            format_op_line(combine.op.name, ptx_type, acc_reg, operands),
        ]
        lane_mask //= 2
    return lines


def emit_tree_halve(tree, reg_alloc: RegisterAllocator, labels: LabelAllocator) -> list[str]:
    """Return the lines that fold the per-warp totals in shared memory into slot 0, then load slot 0 into every thread."""
    # TASK.md: TreeHalve length is a power of two, so halving never drops a value.
    assert tree.length & (tree.length - 1) == 0, "ASSUMPTIONS 22: a TreeHalve length is a power of two"
    assert len(tree.bufs) == len(tree.state) == len(tree.combine_states) == 1, "ASSUMPTIONS 23: a TreeHalve folds one state value"
    assert tree.inner is None and tree.barrier_count is None, "ASSUMPTIONS 23: a TreeHalve uses a flat buffer and a block-wide barrier"
    ptx_type, reg_kind = DTYPE_INFO[tree.dtype.name]
    smem_name = tree.bufs[0]
    # tid_var names the IR variable that is this thread's index into the slots ('warp' here).
    # emit_cooperative_prologue defined it, so the lookup returns the register holding the warp number.
    tid_reg = reg_alloc.lookup(tree.tid_var)
    acc_reg = reg_alloc.lookup(tree.state[0])
    partner_acc_reg = reg_alloc.new_reg(reg_kind)
    # the merge rule refers to the partner's value by this IR name, so it must resolve to partner_acc_reg
    reg_alloc.bind(tree.state_b[0], partner_acc_reg)
    idle_warp_predicate_reg = reg_alloc.new_reg('p')
    combine = tree.combine_states[0]
    operands = [reg_alloc.lookup(arg) for arg in combine.args]
    # the address of my own slot does not change between steps
    slot_address_reg, lines = emit_shared_address(smem_name, Var(name=tree.tid_var), tree.dtype.nbytes, reg_alloc)

    # runs at compile time: one copy of the step's PTX per halving (offset 2, then 1), so the output has no loop
    partner_offset = tree.length // 2
    while partner_offset >= 1:
        end_label = labels.new_label(TREE_END_LABEL)
        partner_byte_offset = partner_offset * tree.dtype.nbytes
        lines += [
            f"{INDENT}{SET_PREDICATE}.{GREATER_EQUALS}{PtxType.U32} {idle_warp_predicate_reg}, {tid_reg}, {partner_offset};",
            f"{INDENT}@{idle_warp_predicate_reg} {BRANCH} {end_label};",
            f"{INDENT}{LOAD}{SHARED}{ptx_type} {acc_reg}, [{slot_address_reg}];",
            f"{INDENT}{LOAD}{SHARED}{ptx_type} {partner_acc_reg}, [{slot_address_reg}+{partner_byte_offset}];",
            format_op_line(combine.op.name, ptx_type, acc_reg, operands),
            f"{INDENT}{STORE}{SHARED}{ptx_type} [{slot_address_reg}], {acc_reg};",
            f"{end_label}:",
            f"{INDENT}{BARRIER_SYNC} {tree.barrier_id};",
        ]
        partner_offset //= 2

    # broadcast: every thread loads the total from slot 0, in place
    root_address_reg = reg_alloc.new_reg('r')
    lines += [
        f"{INDENT}{MOVE}{PtxType.U32} {root_address_reg}, {smem_name};",
        f"{INDENT}{LOAD}{SHARED}{ptx_type} {acc_reg}, [{root_address_reg}];",
    ]
    return lines


def emit_smem(smem, reg_alloc: RegisterAllocator) -> list[str]:
    """Return the .shared declaration for a block-wide array of floats, sized in bytes."""
    assert smem.dtype == "float" and len(smem.extents) == 1, "ASSUMPTIONS 18: an Smem is a 1-D float array"
    reg_alloc.bind_shared(smem.name)
    return [f"{INDENT}{SHARED} {ALIGN} {F32_BYTES} {PtxType.B8} {smem.name}[{smem.extents[0] * F32_BYTES}];"]


def emit_sync(sync) -> list[str]:
    """Return the block-wide barrier: no thread continues until all threads of the block have arrived."""
    assert sync.count is None and not sync.warp, "ASSUMPTIONS 19: a Sync is a block-wide barrier (no count, not warp-only)"
    return [f"{INDENT}{BARRIER_SYNC} {sync.barrier_id};"]


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


def emit_cond(cond, reg_alloc: RegisterAllocator, labels: LabelAllocator) -> list[str]:
    """Return the lines that run the body only for threads where the condition holds."""
    assert not cond.else_body, "ASSUMPTIONS 21: a Cond has no else branch"
    assert isinstance(cond.cond, BinaryExpr) and cond.cond.op == "==", "ASSUMPTIONS 21: a Cond tests one equality"
    left_operand, left_lines = emit_expr(cond.cond.left, reg_alloc)
    right_operand, right_lines = emit_expr(cond.cond.right, reg_alloc)
    skip_predicate_reg = reg_alloc.new_reg('p')
    end_label = labels.new_label(COND_END_LABEL)
    return [
        *left_lines,
        *right_lines,
        f"{INDENT}{SET_PREDICATE}.{NOT_EQUALS}{PtxType.U32} {skip_predicate_reg}, {left_operand}, {right_operand};",
        f"{INDENT}@{skip_predicate_reg} {BRANCH} {end_label};",
        *emit_statements(cond.body, reg_alloc, labels),
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
            case WarpShuffle():
                lines += emit_warp_shuffle(stmt, reg_alloc)
            case Smem():
                lines += emit_smem(stmt, reg_alloc)
            case Cond():
                lines += emit_cond(stmt, reg_alloc, labels)
            case TreeHalve():
                lines += emit_tree_halve(stmt, reg_alloc, labels)
            case Sync():
                lines += emit_sync(stmt)
            case _:
                raise NotImplementedError(f"statement {type(stmt).__name__}")
    return lines
