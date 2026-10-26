"""PTX data types and the mapping from IR dtype names."""
import struct
from enum import StrEnum


class PtxType(StrEnum):
    """PTX type suffixes, spelled as they appear in instructions."""
    U32 = ".u32"
    U64 = ".u64"
    F32 = ".f32"
    B32 = ".b32"
    B64 = ".b64"
    B8 = ".b8"
    PRED = ".pred"


# IR dtype name -> (PTX type, register kind)
DTYPE_INFO = {
    "f32": (PtxType.F32, "f"),
}


def f32_literal(value: float) -> str:
    """Return value as a PTX f32 literal: 0f followed by the 8 hex digits of its IEEE-754 bits."""
    return "0f" + struct.pack(">f", value).hex().upper()
