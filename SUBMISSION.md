# PTX compiler for Emmy Kernel IR

Submission video: https://youtu.be/J5Es1kj2BfY

`compile.py` reads a Kernel IR file and prints PTX to stdout.

```bash
pip install emmy-ml==0.3.11
python compile.py problems/r04_row_sum/k_sum_1_reduce.ir.json > out.ptx
python submit.py r04_row_sum --compiler compile.py     # compile, run on the runner, check the output
```

The compiler also reads the `meta.json` next to the IR file. It supplies the argument names and
order, tensor shapes and the launch geometry (grid and block).

## Tech stack

The compiler needs Python 3.12 and Emmy (`emmy-ml` 0.3.11) for the IR classes. It does not call nvcc.

Used while building it: `nvcc` to generate reference PTX from Emmy's CUDA, `ptxas` to check that
PTX assembles, `submit.py` and the RiftStack runner (RTX 5090) to check results, Claude Code as the
AI assistant, and WSL2, Neovim, Zellij, git and `gh` as the environment.

## Status

| category | problems | status |
|---|---|---|
| 1 pointwise | p00 to p03 | pass |
| 2 reductions | r01 to r04 | pass |
| 3 to 5 (matmul, fused, attention) | m01 to a02 | not implemented |

An unsupported IR statement stops the compiler with `NotImplementedError: statement <Name>`, so a
gap is reported instead of silently miscompiled. Other unsupported input stops at a check (see Assumptions).

## How it works

The compiler is a tree walk over the kernel's `Tile` body. It prints PTX where an interpreter
would evaluate. The body is generated first, because the register counts are only known once every
instruction exists, and is then printed below the `.reg` lines.

| file | job |
|---|---|
| `compile.py` | command line: load the IR and `meta.json`, print the header and the kernels |
| `ptxgen/__init__.py` | `emit_graph`: one `emit_kernel` call per kernel in `meta.json` |
| `ptxgen/kernel.py` | signature, register declarations, parameter loads, the two prologues |
| `ptxgen/statements.py` | one emitter per IR statement (`Load`, `Write`, `Assign`, `Accum`, `StridedLoop`, `Cond`, `WarpShuffle`, `Smem`, `Sync`, `TreeHalve`) |
| `ptxgen/address.py` | index expressions, row-major flattening, global and shared addresses |
| `ptxgen/regs.py` | `RegisterAllocator`: register names, IR name to register, tensor base address and shape, shared arrays |
| `ptxgen/labels.py` | unique branch label names |
| `ptxgen/consts.py`, `ptxgen/types.py` | PTX spellings, the dtype table, f32 literals |

## IR to PTX

| IR | PTX |
|---|---|
| scalar `Tile` | `gid = ctaid.x * ntid.x + tid.x`, then `setp.ge` and `bra DONE` for out-of-range threads |
| cooperative `Tile` | last axis is `tid.x`, an axis before it is `ctaid.x` (one block per row); `lane` and `warp` derived from `tid.x` |
| `Load`, `Write` | `ld.global`, `st.global` (vector form for 2 or 4 values); index flattened row-major from the tensor shape |
| `Assign` | `add`, `mul`, `max`, `div.rn` |
| `StridedLoop` + `Accum` | counter register, accumulator seeded before the loop label, `add` written back in place |
| `WarpShuffle` | `shfl.sync.bfly.b32` ladder, lane mask 16 down to 1 |
| `Smem`, `Sync` | `.shared` declaration in place, `bar.sync` |
| `Cond` | `setp.ne` and a forward branch past the body |
| `TreeHalve` | per-step `setp.ge` skip, shared loads, merge, shared store, `bar.sync`; then every thread reloads slot 0 |

## Assumptions

`ASSUMPTIONS.md` lists 24 rows about what the compiler expects from its input. Each row names the
code that relies on it and how it is checked. Most checks are `assert` statements whose message
starts with the row number. A few raise `KeyError` or `TypeError` instead. Rows 3 and 13 cannot be
checked at compile time and are documented only.

## Design notes

- Output is `.version 8.0` and `.target sm_89`, as the task specifies.
- `shfl.sync.bfly.b32` takes the float registers directly, as Emmy's reference CUDA does
  (`__shfl_xor_sync` on a float). The worked example in `examples/` round-trips through integer
  registers with `mov.b32`; I checked that `ptxas` accepts the direct form.
- `div.f32` needs a rounding modifier in PTX, so divide is `div.rn.f32` (IEEE round to nearest).
- `TreeHalve` follows the IR contract that its length is a power of two.
- Per-kernel state (registers, labels) is passed as two arguments. One `KernelContext` object is
  the cleaner design; it was left for later because of the deadline.

## Known limits

f32 only. Launches use only the x dimension (ASSUMPTIONS 7). At most one row axis in a cooperative
tile. `TreeHalve` folds one value. `Cond` tests one equality with no else branch. No instruction
scheduling or register reuse beyond keeping accumulators in place.

## AI assistance

I used Claude Code as a tutor and as the typist. I wrote the first working PTX for p00 by hand, and
the first part of `compile.py` (header and signature) myself, so that I understood the PTX syntax
and knew what the compiler had to produce before anything was generated for me. After that I decided
each design step and the assistant typed small changes that I approved and committed. It also
explained PTX and the Emmy IR, quizzed me on the pieces, and helped draft these docs. I can explain
every function, at the time of writing this 😅😅
