"""Index and address math: turn IR index expressions into PTX operands and addresses."""

from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from ptxgen.consts import ADD, BINARY_OPS, INDENT, MOVE, MULTIPLY_WIDE
from ptxgen.regs import RegisterAllocator
from ptxgen.types import PtxType


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
            if op == "+" and right_operand == "0":
                return left_operand, left_lines
            result_reg = reg_alloc.new_reg('r')
            instruction = BINARY_OPS[op]
            line = f"{INDENT}{instruction}{PtxType.U32} {result_reg}, {left_operand}, {right_operand};"
            # Post-order: the operands' lines come before the instruction that reads them.
            return result_reg, [*left_lines, *right_lines, line]
        case _:
            raise NotImplementedError(f"expression {type(expr).__name__}")


def emit_flat_index(tensor_name, index_exprs, reg_alloc):
    """Return (operand, lines) for the flat position of a multi-dimensional index, row-major."""
    shape = reg_alloc.lookup_shape(tensor_name)
    assert len(index_exprs) == len(shape), "ASSUMPTIONS 10: an index has one expression per tensor dimension"
    flat_operand, lines = emit_expr(index_exprs[0], reg_alloc)
    # extra indices (2-D and up): position = position * next dimension's length + next index
    for expr, extent in zip(index_exprs[1:], shape[1:]):
        next_operand, next_lines = emit_expr(expr, reg_alloc)
        scaled_reg = reg_alloc.new_reg('r')
        result_reg = reg_alloc.new_reg('r')
        lines += [
            *next_lines,
            f"{INDENT}{BINARY_OPS['*']}{PtxType.U32} {scaled_reg}, {flat_operand}, {extent};",
            f"{INDENT}{ADD}{PtxType.U32} {result_reg}, {scaled_reg}, {next_operand};",
        ]
        flat_operand = result_reg
    return flat_operand, lines


def emit_element_address(tensor_name, index_exprs, bytes_per_element, reg_alloc):
    """Return (address_reg, lines) for the element of a tensor at index_exprs."""
    index_operand, index_lines = emit_flat_index(tensor_name, index_exprs, reg_alloc)
    base_reg = reg_alloc.lookup_tensor(tensor_name)
    if index_operand == "0":
        return base_reg, index_lines
    # address = base address of the tensor + byte offset of the element
    offset_reg = reg_alloc.new_reg('rd')
    address_reg = reg_alloc.new_reg('rd')
    return address_reg, [
        *index_lines,
        f"{INDENT}{MULTIPLY_WIDE}{PtxType.U32} {offset_reg}, {index_operand}, {bytes_per_element};",
        f"{INDENT}{ADD}{PtxType.U64} {address_reg}, {base_reg}, {offset_reg};",
    ]


def emit_shared_address(array_name, index_expr, bytes_per_element, reg_alloc):
    """Return (address_reg, lines) for element index_expr of a shared array."""
    index_operand, index_lines = emit_expr(index_expr, reg_alloc)
    base_reg = reg_alloc.new_reg('r')
    offset_reg = reg_alloc.new_reg('r')
    address_reg = reg_alloc.new_reg('r')
    return address_reg, [
        *index_lines,
        f"{INDENT}{MOVE}{PtxType.U32} {base_reg}, {array_name};",
        f"{INDENT}{BINARY_OPS['*']}{PtxType.U32} {offset_reg}, {index_operand}, {bytes_per_element};",
        f"{INDENT}{ADD}{PtxType.U32} {address_reg}, {base_reg}, {offset_reg};",
    ]
