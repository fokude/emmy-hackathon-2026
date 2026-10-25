"""Register naming and bookkeeping for one kernel."""

from collections import defaultdict
from enum import StrEnum

class RegType(StrEnum):
    """Register kinds, valued by their PTX name prefix (%f, %p, %r, %rd)."""
    REG_TYPE_FLOAT = 'f'
    REG_TYPE_PREDICATE = 'p'
    REG_TYPE_INT32 = 'r'
    REG_TYPE_INT64 = 'rd'

VALID_KINDS = [t.value for t in RegType]

class RegisterAllocator:
    """Hands out register names for one kernel and tracks what each holds.

    Also maps each IR value name to the register that holds it.
    """

    def __init__(self):
        self.counter = defaultdict(int)
        self.value_name_to_reg: dict[str, str] = {}  # value name (Load/Assign result) -> register name

    def new_reg(self, kind: str) -> str:
        """Return a fresh register name of the given kind, such as %f0 then %f1."""
        try:
            reg_type = RegType(kind)
        except ValueError:
            raise ValueError(f"Invalid register type '{kind}'. Expected one of {VALID_KINDS}")

        reg_id = self.counter[reg_type]
        self.counter[reg_type] += 1

        return f"%{reg_type.value}{reg_id}"

    def bind(self, value_name: str, reg: str) -> None:
        """Record that the IR value value_name lives in register reg."""
        self.value_name_to_reg[value_name] = reg

    def lookup(self, value_name: str) -> str:
        """Return the register holding value_name, or raise KeyError if undefined."""
        if value_name not in self.value_name_to_reg:
            raise KeyError(f"IR value '{value_name}' used before it was defined")
        return self.value_name_to_reg[value_name]

    def counts(self) -> dict[str, int]:
        """Return how many registers of each kind were handed out."""
        return {reg_type.value: n for reg_type, n in self.counter.items()}
