"""PTX data types and the mapping from IR dtype names."""
from enum import StrEnum


class PtxType(StrEnum):
    """PTX type suffixes, spelled as they appear in instructions."""
    U32 = ".u32"
    U64 = ".u64"
    F32 = ".f32"
    B32 = ".b32"
    B64 = ".b64"
    PRED = ".pred"


# IR dtype name -> (PTX type, register kind)
DTYPE_INFO = {
    "f32": (PtxType.F32, "f"),
}
