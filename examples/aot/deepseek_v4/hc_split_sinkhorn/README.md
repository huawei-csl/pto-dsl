# hc_split_sinkhorn — PTO DSL port

Original (TileLang/GPU): produces `(pre, post, comb)` per row of `mixes`,
where `pre`/`post` are scaled-sigmoid heads and `comb` is sinkhorn-normalized
(row-softmax then alternating row/col normalization for `sinkhorn_iters`
rounds).

## Status

Full op runs **on-device** in a single `pto.vector_section`. Nothing
remains on the host except the per-test driver code.

Per sample (HC=4, MIX_HC=24):
- `pre  = sigmoid(mixes[:, :HC]*scale[0] + base[:HC]) + eps`
- `post = 2 * sigmoid(mixes[:, HC:2HC]*scale[1] + base[HC:2HC])`
- `comb = sinkhorn(mixes[:, 2HC:].reshape(HC,HC) * scale[2] + base[2HC:])`

Sigmoid is composed as `tile.muls(-1) → tile.exp → tile.adds(1) →
tile.reciprocal` (note: `pto.trecip` requires distinct src/dst, so a
scratch subview is used). The three scalar scales are read once via
`pto.load_scalar`; the three base sub-tensors are loaded into VEC tiles
once per worker and broadcast/added with `tile.add`.

## Dtype mapping
All FP32 — matches GPU exactly.
