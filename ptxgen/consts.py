VISIBLE = ".visible"
ENTRY = ".entry"
PARAM_DECLARATION = ".param"
PARAM_PREFIX = PARAM_DECLARATION[1:] + "_"

U64_TYPE = ".u64"
U32_TYPE = ".u32"
TAB_SPACE = 2
SPACE_CHAR = ' '

# register kind (see regs.RegType) -> type used in the .reg declaration
REG_DECL_TYPES = {
    "f": ".f32",
    "p": ".pred",
    "r": ".b32",
    "rd": ".b64",
}

MOVE = "mov"
MULTIPLY_ADD = "mad"
LOWER_BITS = "lo"
DONE_LABEL = "DONE"
SET_PREDICATE = "setp"
GREATER_EQUALS = "ge"
BRANCH = "bra"
