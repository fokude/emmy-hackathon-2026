# Assumptions

What the compiler assumes about its input. Each one is checked with an `assert` or a clear error
where the code relies on it, so a violation fails loudly instead of producing wrong PTX.

| # | Assumption | Where it is relied on | Checked |
|---|---|---|---|
| 1 | `meta.json` sits next to the IR file and lists every kernel to compile | `compile.py` `parse_args` | file read errors exit with a message |
| 2 | Each `args` entry in `meta.json` has a name listed in `arg_order` | `ptxgen/emit.py` `emit_params` | not yet asserted |
| 3 | Every kernel argument is a buffer pointer, declared `.u64` | `emit_params` | not yet asserted |
| 4 | A kernel node's body starts with one `Tile` | `emit_kernel_body` | not yet asserted |
| 5 | A `Tile` has exactly one axis | `emit_prologue` | `assert len(tile.axes) == 1` |
| 6 | The axis extent is a static integer | `emit_prologue` | `Dim.as_static()` raises `TypeError` |
| 7 | The launch is one-dimensional (`ctaid.x`, `ntid.x`, `tid.x`) | `emit_prologue` | not yet asserted |
| 8 | Every dtype is `f32` | statement emitters (not written yet) | to be asserted |
| 9 | The tile is the scalar tier (`block_threads` is `None`), so the axis variable equals the global thread id | `emit_prologue` | not yet asserted |
| 10 | `Load` and `Write` indices have exactly one expression (1-D tensors) | `emit_load`, `emit_write` | `assert len(load.index) == 1` in `emit_load` (`emit_write` likewise) |
| 11 | A `Load` defines exactly one value (no vector loads) | `emit_load` | `assert len(load.names) == 1` |
| 12 | A `Write` stores 2 or 4 values (vector store, no scalar store) | `emit_write` | `assert len(write.values) in (2, 4)` |
