# Assumptions

What the compiler assumes about its input. Each one is checked with an `assert` or a clear error
where the code relies on it, so a violation fails loudly instead of producing wrong PTX.

| # | Assumption | Where it is relied on | Checked |
|---|---|---|---|
| 1 | `meta.json` sits next to the IR file and lists every kernel to compile | `compile.py` `parse_args` | file read errors exit with a message |
| 2 | Each `args` entry in `meta.json` has a name listed in `arg_order` | `ptxgen/emit.py` `emit_params` | `assert set(arg_dict) == set(arg_names)` |
| 3 | Every kernel argument is a buffer pointer, declared `.u64` | `emit_params` | not checkable: `meta.json` has no scalar marker |
| 4 | A kernel node's body starts with one `Tile` | `emit_kernel_body` | `assert len(body) == 1 and isinstance(body[0], Tile)` |
| 5 | A scalar `Tile` has exactly one axis. A cooperative `Tile` has one axis, or a row axis followed by the cooperative axis | `emit_prologue`, `emit_cooperative_prologue` | `assert` on `len(tile.axes)` in each |
| 6 | The axis extent is a static integer | `emit_prologue` | `Dim.as_static()` raises `TypeError` |
| 7 | The launch is one-dimensional (`ctaid.x`, `ntid.x`, `tid.x`) | `emit_prologue` | `assert` on `grid` and `block` in `emit_kernel` |
| 8 | Every dtype is `f32` | `emit_load`, `emit_assign`, `emit_write` | `DTYPE_INFO[...]` raises `KeyError` on any other dtype |
| 9 | A tile with `block_threads` `None` is the scalar tier, so the axis variable equals the global thread id | `emit_prologue` | `emit_kernel_body` picks the prologue from `tile.block_threads` |
| 10 | A `Load` or global `Write` index has one expression per tensor dimension (row-major layout); a `Write` to a single-element tensor may have none; a shared `Write` has exactly one | `emit_flat_index`, `emit_global_write`, `emit_shared_write` | `assert` on `len(index)` in `emit_flat_index` and `emit_shared_write` |
| 11 | A `Load` defines exactly one value (no vector loads) | `emit_load` | `assert len(load.names) == 1` |
| 12 | A global `Write` stores 1, 2 or 4 values (scalar or vector store) | `emit_global_write` | `assert len(write.values) in (1, 2, 4)` |
| 13 | Inputs contain no NaN, because `max.f32` returns the non-NaN operand so `relu(NaN)` becomes 0 where PyTorch keeps NaN | `ELEMENTWISE_OPS` (`relu`) | not checkable at compile time |
| 14 | A cooperative tile launches one block per row-axis value (one block if there is no row axis), and the block size equals the cooperative axis extent | `emit_cooperative_prologue` | `assert` on `grid` and `block` |
| 15 | A `StridedLoop` stops at its axis extent (`end` is `None`) | `emit_strided_loop` | `assert loop.end is None` |
| 16 | A `StridedLoop` has `seed` True | `emit_accum_seeds` | `assert loop.seed` |
| 17 | A `WarpShuffle` spans one full warp (`length` 32) and combines one state value | `emit_warp_shuffle` | `assert` on `length`, `state` and `combine_states` |
| 18 | An `Smem` is a 1-D float array | `emit_smem` | `assert` on `dtype` and `extents` |
| 19 | A `Sync` is a block-wide barrier (no `count`, not warp-only) | `emit_sync` | `assert` on `count` and `warp` |
| 20 | A shared `Write` stores one value | `emit_shared_write` | `assert len(write.values) == 1` |
| 21 | A `Cond` tests one equality and has no else branch | `emit_cond` | `assert` on `else_body` and `cond.op` |
| 22 | A `TreeHalve` length is a power of two (guaranteed by the IR contract in TASK.md) | `emit_tree_halve` | `assert (length & (length - 1)) == 0` |
| 23 | A `TreeHalve` folds one state value in a flat buffer with a block-wide barrier | `emit_tree_halve` | `assert` on `bufs`, `state`, `combine_states`, `inner`, `barrier_count` |
| 24 | The IR's `lane` and `warp` variables are the thread's position in its warp and its warp number, which the compiler must define (they are not in the IR) | `emit_cooperative_prologue` | `reg_alloc.lookup` raises `KeyError` when a name is missing |
