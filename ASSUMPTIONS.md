# Assumptions

What the compiler assumes about its input. Each one is checked with an `assert` or a clear error
where the code relies on it, so a violation fails loudly instead of producing wrong PTX.

| # | Assumption | Where it is relied on | Checked |
|---|---|---|---|
| 1 | `meta.json` sits next to the IR file and lists every kernel to compile | `compile.py` `parse_args` | file read errors exit with a message |
| 2 | Each `args` entry in `meta.json` has a name listed in `arg_order` | `ptxgen/emit.py` `emit_params` | `assert set(arg_dict) == set(arg_names)` |
| 3 | Every kernel argument is a buffer pointer, declared `.u64` | `emit_params` | not checkable: `meta.json` has no scalar marker |
| 4 | A kernel node's body starts with one `Tile` | `emit_kernel_body` | `assert len(body) == 1 and isinstance(body[0], Tile)` |
| 5 | A `Tile` has exactly one axis | `emit_prologue` | `assert len(tile.axes) == 1` |
| 6 | The axis extent is a static integer | `emit_prologue` | `Dim.as_static()` raises `TypeError` |
| 7 | The launch is one-dimensional (`ctaid.x`, `ntid.x`, `tid.x`) | `emit_prologue` | `assert` on `grid` and `block` in `emit_kernel` |
| 8 | Every dtype is `f32` | `emit_load`, `emit_assign`, `emit_write` | `DTYPE_INFO[...]` raises `KeyError` on any other dtype |
| 9 | A tile with `block_threads` `None` is the scalar tier, so the axis variable equals the global thread id | `emit_prologue` | `emit_kernel_body` picks the prologue from `tile.block_threads` |
| 10 | `Load` and `Write` indices have exactly one expression (1-D tensors) | `emit_load`, `emit_write` | `assert len(load.index) == 1` in `emit_load` (`emit_write` likewise) |
| 11 | A `Load` defines exactly one value (no vector loads) | `emit_load` | `assert len(load.names) == 1` |
| 12 | A `Write` stores 2 or 4 values (vector store, no scalar store) | `emit_write` | `assert len(write.values) in (2, 4)` |
| 13 | Inputs contain no NaN, because `max.f32` returns the non-NaN operand so `relu(NaN)` becomes 0 where PyTorch keeps NaN | `ELEMENTWISE_OPS` (`relu`) | not checkable at compile time |
| 14 | Cooperative tiles are supported only in the r01 to r03 shape: one axis, one block, block size equals the axis extent (r04 adds a `ctaid.x` row axis later) | `emit_cooperative_prologue` | `assert` on `grid` and `block` |
| 15 | A `StridedLoop` stops at its axis extent (`end` is `None`) | `emit_strided_loop` | `assert loop.end is None` |
