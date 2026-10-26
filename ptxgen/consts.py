"""PTX keywords and formatting constants, grouped by category."""
from ptxgen.types import PtxType

# --- directives ---
VISIBLE = ".visible"
ENTRY = ".entry"
PARAM = ".param"
GLOBAL = ".global"
SHARED = ".shared"
ALIGN = ".align"

# --- types ---

# register kind (see regs.RegType) -> type used in the .reg declaration
REG_DECL_TYPES = {
    "f": PtxType.F32,
    "p": PtxType.PRED,
    "r": PtxType.B32,
    "rd": PtxType.B64,
}

# --- operations ---
MOVE = "mov"
LOAD = "ld"
STORE = "st"
CONVERT_ADDRESS = "cvta"

# IR index operator -> PTX instruction
BINARY_OPS = {"+": "add", "*": "mul.lo"}
MULTIPLY_WIDE = "mul.wide"
ADD = "add"

# IR elementwise op name -> (PTX instruction, extra constant operands)
ELEMENTWISE_OPS = {
    "add": ("add", []),
    "multiply": ("mul", []),
    "relu": ("max", ["0f00000000"]),
}
MULTIPLY_ADD = "mad"
SET_PREDICATE = "setp"
BRANCH = "bra"
AND = "and"
SHIFT_RIGHT = "shr"
BARRIER_SYNC = "bar.sync"

# shfl.sync.bfly: lane i reads the register of lane (i XOR lane_mask), within the same warp
SHUFFLE_BUTTERFLY = "shfl.sync.bfly"
# shfl operand c: lanes may read from lane numbers 0 to this value (0x1f = 31, the whole warp), so no read is blocked.
# PTX manual name: clamp.
SHUFFLE_MAX_LANE = "0x1f"
# bit i set = lane i must take part in the shuffle; all 32 bits set = the full warp
FULL_WARP_MASK = "0xffffffff"
# lanes (threads) per warp
WARP_SIZE = 32
# bytes per float
F32_BYTES = 4

# --- operation modifiers ---
LOWER_BITS = "lo"
GREATER_EQUALS = "ge"

# --- symbol names (identifiers we create in the PTX) ---
PARAM_PREFIX = PARAM[1:] + "_"
DONE_LABEL = "DONE"
LOOP_LABEL = "LOOP"
LOOP_END_LABEL = "LOOP_END"

# --- IR variable names ---
LANE_VAR = "lane"
WARP_VAR = "warp"

# --- formatting ---
TAB_SPACE = 2
SPACE_CHAR = ' '
INDENT = SPACE_CHAR * TAB_SPACE
