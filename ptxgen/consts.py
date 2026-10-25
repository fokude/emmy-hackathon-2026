"""PTX keywords and formatting constants, grouped by category."""

# --- directives ---
VISIBLE = ".visible"
ENTRY = ".entry"
PARAM = ".param"

# --- types ---
U64_TYPE = ".u64"
U32_TYPE = ".u32"

# register kind (see regs.RegType) -> type used in the .reg declaration
REG_DECL_TYPES = {
    "f": ".f32",
    "p": ".pred",
    "r": ".b32",
    "rd": ".b64",
}

# --- operations ---
MOVE = "mov"
LOAD = "ld"
CONVERT_ADDRESS = "cvta"

# IR index operator -> PTX instruction
BINARY_OPS = {"+": "add", "*": "mul.lo"}
MULTIPLY_ADD = "mad"
SET_PREDICATE = "setp"
BRANCH = "bra"

# --- operation modifiers ---
LOWER_BITS = "lo"
GREATER_EQUALS = "ge"

# --- symbol names (identifiers we create in the PTX) ---
PARAM_PREFIX = PARAM[1:] + "_"
DONE_LABEL = "DONE"

# --- formatting ---
TAB_SPACE = 2
SPACE_CHAR = ' '
INDENT = SPACE_CHAR * TAB_SPACE
