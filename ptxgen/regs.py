from collections import defaultdict
from enum import StrEnum

class RegType(StrEnum):
    REG_TYPE_FLOAT = 'f'
    REG_TYPE_PREDICATE = 'p'
    REG_TYPE_INT32 = 'r'
    REG_TYPE_INT64 = 'rd'

VALID_KINDS = [t.value for t in RegType]

class RegisterAllocator:
    def __init__(self):
        self.counter = defaultdict(int)
        self.value_name_to_reg: dict[str, str] = {}  # value name (Load/Assign result) -> register name

    def new_reg(self, kind: str) -> str:
        try:
            reg_type = RegType(kind)
        except ValueError:
            raise ValueError(f"Invalid register type '{kind}'. Expected one of {VALID_KINDS}")

        reg_id = self.counter[reg_type]
        self.counter[reg_type] += 1

        return f"%{reg_type.value}{reg_id}"

    def bind(self, value_name: str, reg: str) -> None:
        self.value_name_to_reg[value_name] = reg

    def lookup(self, value_name: str) -> str:
        if value_name not in self.value_name_to_reg:
            raise KeyError(f"IR value '{value_name}' used before it was defined")
        return self.value_name_to_reg[value_name]

    def counts(self) -> dict[str, int]:
        return {reg_type.value: n for reg_type, n in self.counter.items()}
