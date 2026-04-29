# DSL split_pipe kernel — fix backlog (vs `cpp_ref/split_pipe`)

Goal: **`python run.py`** matches **`../cpp_ref/split_pipe/run.py`** for **correctness** (torch fp32 reference) and **performance** (TFLOP/s in the same ballpark as JIT `fa_performance_kernel.cpp`) on NPU.

---

## Resolved (in builder / workflow)

- [x] **A1 — VEC UB overflow from pipe `tpop` staging** — stacked layout under 192 KiB; `_VEC_UB_TAIL` assert.
- [x] **A2 — Cube RIGHT bank overflow (`HEAD=128`, `S1_TILE=512`)** — share K/V RIGHT at `RIGHT_KV_OFF=0`.
- [x] **A3 — Duplicate UB offset for `p_fp32` / `p_fp16`** — `VEC_P_FP16_OFF = VEC_P_FP32_OFF + _TILE_FP16_BYTES`; distinct `TASSIGN` in generated vec code.
- [x] **B3 — Rebuild `FA_TILES` variants** — `bash compile.sh` produces `fa.so` … `fa_128.so`.
- [x] **A9 — Compile/runtime env documentation** — `FA_Q_ROWS`, `FA_S1_TILE` wired through `compile.sh`; documented in `README.md`.
- [x] **A12 — Host/builder constant drift (NaNs from mismatched `FA_*`)** — `compile.sh` emits **`build_artifacts/fa${TAG}.build_env`** per variant; **`run.py`** applies the sidecar whose **`FA_NUM_TILES * FA_S1_TILE`** equals the first **`FA_BENCH_LENGTHS`** entry (tie-break: **`FA_S1_TILE`** env if set, else **`fa.build_env`**), then **`importlib.reload(fa_performance_builder)`** so GM/tensor sizes match the loaded `.so`.

---

## Open — blocking default port

- [ ] **A7 — CCU / aicore fault at default `S1_TILE=512`**  
      **`python run.py`** (default env aligned with `fa.so` + **A12**) still fails at **`torch.npu.synchronize()`** with **ACL 507015**, **CCU instruction address check** on some devices. **`../cpp_ref/split_pipe`** JIT passes on the **same** hardware. Needs bisheng/ptotas or vendor triage against **`build_artifacts/fa.cpp`**.

- [ ] **A11 — Numerics on alternate tile width**  
      With **`FA_S1_TILE=256`** and **matching** compile/runtime **`FA_NUM_TILES` / `FA_Q_ROWS`**, the kernel **may run** without the A7 fault but **`torch.testing.assert_close`** can still fail (**NaNs** / large drift). Treat as schedule vs reference macro alignment or toolchain issue separate from A7.

---

## Open — parity / audit (non-blocking for “runs at all”)

- [ ] **A4 — Bisheng stack / spill** — stack `0x8000`; larger stacks rejected by the toolchain in this environment; not proved root cause of A7.

- [ ] **A5 — Launch geometry** — reference uses `runTFA<<<S0/CUBE_S0,…>>>`, DSL uses `min(NUM_Q_BLOCKS, cores)` striping; totals consistent; optional CV audit.

- [ ] **A6 — Structural parity** — reference: fused **`runTFA`**, **`QK_PRELOAD=4`**, **`CUBE_S0=128`**; DSL: **`pto.call(cube); pto.call(vec)`**, **`QK_PRELOAD=2`**, **`S0=32`**. Reference constructs **`TPipe`** in **QK → P → PV** order (`BUF0_QK_READY` / `BUF1_SM_READY` / `UPDATE_READY`); DSL initializes **QK → PV → P** (two **`l2g2l_pipe`** then V2C **`aic/aiv_initialize_pipe`**). Numerical match is the target once A7/A11 clear.

- [ ] **A8 — CV tail sync vs `runTFA`** — reference ends cube with **`wait_flag_dev(CV_BLOCK_END)`** and vec with **`ffts_cross_core_sync(..., CV_BLOCK_END)`**; DSL output ends with **`ptoas_auto_sync_tail`**. May require future **`pto`/ptoas** hooks if parity demands explicit FFTS tail patterns.

---

## Verification checklist

- [ ] **B1 — `python run.py`** passes **`torch.testing.assert_close`** vs **`fa_reference`** for default + full **`FA_BENCH_LENGTHS`**. **Blocked by A7 / possibly A11** until NPU/toolchain issues clear.

- [ ] **B2 — Benchmark vs `cpp_ref/split_pipe/run.py`** (`jit_compile_flash`), same FLOP model (`attn_flops_matmul_softmax_scale`), `flush_cache=False`. **Blocked by A7** for apples-to-apples default geometry.

---

### Status log

| Date | Change |
| ---- | ------ |
| (session) | Backlog created; A3; NPU CCU fault |
| 2026-04-29 | A3 verified in `fa.cpp`; `compile.sh`: **FA_Q_ROWS**, **FA_S1_TILE**; README status; default **512** path **A7**; **256** path **A11** |
| 2026-04-29 | **A12**: **`fa*.build_env`** + **`run.py`** **`reload`**; README pipe-order wording corrected (**QK→PV→P** vs ref **QK→P→PV**); removed obsolete “**flag_base**” resolved line |
