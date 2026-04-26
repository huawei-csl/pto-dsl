# fp4_act_quant — PTO DSL port

Original (TileLang/GPU): BF16 → FP4(e2m1) per-block quant, block_size=32,
FP4_max=6, power-of-2 E8M0 scale.

PTO DSL has no FP4/BF16/E8M0 native support, so this port uses
FP16 → int8 (range ±7) with FP32 scale = amax / 6. Algorithm and shapes
mirror the original; round-to-power-of-2 scale and packed FP4 storage are
omitted.
