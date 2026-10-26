"""Generate PTX from Emmy Kernel IR."""

from emmy.compiler.graph import Graph
from ptxgen.kernel import emit_kernel

__all__ = ["emit_graph"]


def emit_graph(graph: Graph, metadata: dict) -> list[str]:
    """Return the PTX lines for every kernel listed in the metadata."""
    lines = []
    for kernel in metadata["kernels"]:
        lines += emit_kernel(graph, kernel)
    return lines
