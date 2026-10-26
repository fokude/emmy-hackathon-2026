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


def emit_element_address(tensor_name, index_expr, bytes_per_element, reg_alloc):
    """Return (address_reg, lines) for element index_expr of a tensor."""
    index_operand, index_lines = emit_expr(index_expr, reg_alloc)
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
