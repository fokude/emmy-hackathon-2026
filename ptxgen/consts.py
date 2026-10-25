# --- directives ---
VISIBLE = ".visible"
ENTRY = ".entry"
PARAM_DECLARATION = ".param"

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
MULTIPLY_ADD = "mad"
SET_PREDICATE = "setp"
BRANCH = "bra"

# --- operation modifiers ---
LOWER_BITS = "lo"
GREATER_EQUALS = "ge"

# --- symbol names (identifiers we create in the PTX) ---
PARAM_PREFIX = PARAM_DECLARATION[1:] + "_"
DONE_LABEL = "DONE"

# --- formatting ---
TAB_SPACE = 2
SPACE_CHAR = ' '
