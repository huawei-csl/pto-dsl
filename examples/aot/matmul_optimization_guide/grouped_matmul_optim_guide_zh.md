# 从零把 Matmul 泛化到 Grouped Matmul
(基于 PTO-DSL 的逐步优化指南)

本文是 [`mamtul_optim_guide_zh.md`](https://github.com/huawei-csl/pto-dsl/blob/main/examples/aot/matmul_optimization_guide/mamtul_optim_guide_zh.md) 的配套扩展，目标不是重复讲一遍 Matmul，而是说明:

- 什么时候 plain matmul 的优化套路可以直接复用
- 什么时候必须引入新的 group 级调度
- 如何在 PTO-DSL 里把 `matmul -> grouped_matmul -> grouped_matmul_add` 这条链路写成同一套 tile-first 思路

文中的 PTO-DSL 代码风格对齐本地 `examples/aot/matmul_swizzle/step_by_step_guide/optimization_guide.md`，而 grouped matmul 的 kernel 结构参考当前迁移中的 PTO kernel:

- `pto-kernels/python/pto_kernels/ops/gmm/grouped_matmul/kernel.py`
- `pto-kernels/python/pto_kernels/ops/gmm/grouped_matmul_add/kernel.py`

日期: 2026/03/23

# 目录

- 写作目标
- 第 0 步: plain matmul 到 grouped matmul，真正多了什么
- 第 1 步: 功能正确的 grouped matmul baseline
- 第 2 步: 先做 group 内 tile 调度，再谈 group 间负载均衡
- 第 3 步: 复用 swizzle，而不是重写一套新 schedule
- 第 4 步: 从 grouped matmul 扩展到 grouped matmul add
- 第 5 步: 真正值得继续优化的方向
- 附录 A: 一个最小 PTO-DSL grouped matmul 骨架
- 附录 B: grouped matmul 调参 checklist

# 写作目标

plain matmul 和 grouped matmul 的关系，最容易讲歪的地方是:

- 有人把 grouped matmul 当成“多次 matmul 简单 for-loop 拼起来”
- 有人把 grouped matmul 当成“必须完全重写一套 kernel”

这两种理解都不对。

更准确地说:

- **计算核心没变**: 每个 group 内部还是 `A_g @ B_g`
- **优化目标没变**: 还是想让 `GM -> L1 -> L0 -> Cube -> GM` 这条路径尽量满
- **真正新增的是调度维度**: 现在 tile 空间从 `(m_idx, n_idx, k_idx)` 扩展成 `(g_idx, m_idx, n_idx, k_idx)`

所以 grouped matmul 最自然的做法不是发明新算子语义，而是:

1. 先把单 group 的 matmul tile 方案固定住
2. 再决定“group 维度和 MN tile 维度怎么一起分配给 core”
3. 最后再决定 epilogue 是否融合

# 第 0 步: plain matmul 到 grouped matmul，真正多了什么

先回忆 plain matmul:

```text
C[M, N] = A[M, K] @ B[K, N]
```

tile 化以后，顶层常见循环是:

```text
for li in range(core_loop):
    m_idx, n_idx = schedule(li)
    for k_idx in range(k_iters):
        load A_tile
        load B_tile
        matmul / matmul_acc
```

grouped matmul 把上式扩成:

```text
for g in range(G):
    C[g, M, N] = A[g, M, K] @ B[g, K, N]
```

于是 tile 空间多了一个 `g`:

```text
for logical_block in range(total_group_tiles):
    g_idx = logical_block // tiles_per_group
    local_tile = logical_block % tiles_per_group
    m_idx, n_idx = schedule(local_tile)
    for k_idx in range(k_iters):
        load A[g_idx, ...]
        load B[g_idx, ...]
        matmul / matmul_acc
```

注意这里最关键的一点:

- **K 维分块逻辑不需要因为 grouped 而改变**
- **Cube tile 的形状也通常不需要因为 grouped 而改变**
- 变化集中在:
  - 顶层工作分发
  - group 内外 tile 遍历顺序
  - group 粒度不均匀时的负载均衡

如果每个 group 的 `(M, K, N)` 相同，那么 grouped matmul 最适合先做成:

- `group 维是外层逻辑索引`
- `每个 group 内仍沿用 plain matmul 的 swizzle / tile 策略`

这也是当前 PTO grouped matmul seed 的结构。

# 第 1 步: 功能正确的 grouped matmul baseline

## 1.1 先固定 tile，而不是先纠结 group_list

对于一个规则的 dense grouped matmul seed，可以先用统一 shape:

- `A: [G, M, K]`
- `B: [G, K, N]`
- `C: [G, M, N]`

或者像当前迁移中的简化 slice 一样，先固定到单 group dense 版本，再把同一套 block 策略推广到多 group。

无论是哪种写法，baseline 的核心都应该保持和 plain matmul 一致的 cube 配置。例如:

```python
base_m = 16
base_k = 64
base_n = 64
```

这意味着:

- `A_tile = [base_m, base_k]`
- `B_tile = [base_k, base_n]`
- `C_tile = [base_m, base_n]`

grouped 不会改变这些 tile 的语义，只会改变这些 tile 指向哪个 group。

## 1.2 顶层逻辑索引怎么拆

最直接的写法是先把所有 group 的 MN tiles 打平:

```python
tiles_per_group = m_tiles * n_tiles
total_group_tiles = group_count * tiles_per_group
```

然后在 kernel 里:

```python
for logical_block in range(bid, total_group_tiles, num_blocks):
    g_idx = logical_block // tiles_per_group
    local_tile = logical_block % tiles_per_group
    m_idx = local_tile // n_tiles
    n_idx = local_tile % n_tiles
```

这一步的价值在于:

- group 只是“地址基址”的额外偏移
- 单 group 内部的 matmul 微结构完全不变
- 代码结构上非常接近 plain matmul baseline，便于对照验证

## 1.3 PTO-DSL 里需要变的只有 view 偏移

如果 plain matmul 的 `slice_view` 是:

```python
sv_a = pto.slice_view(view_a, source=tv_a, offsets=[m_off, k_off], sizes=[cTileM, cBaseK])
sv_b = pto.slice_view(view_b, source=tv_b, offsets=[k_off, n_off], sizes=[cBaseK, cTileN])
```

那么 grouped 版本本质只是多一层 group 基址:

```python
group_a_off = g_idx * group_a_stride
group_b_off = g_idx * group_b_stride
```

如果张量真的是 rank-3，也可以直接把 `g_idx` 放进 `as_tensor` 的 shape / strides 设计里。关键不是 rank-2 还是 rank-3，而是:

- **每个 group 内的 tile 访问必须仍然规整**
- **不要把 group 逻辑写成大量 scalar 地址拼接**

grouped matmul 写差了，最常见的问题不是 matmul 本身，而是上层调度把 kernel 写成“多一层 scalar for-loop 包着很多小 matmul”。那样很容易:

- 核间负载不均
- group 之间 cache 局部性差
- hot path 退化成标量调度代码

# 第 2 步: 先做 group 内 tile 调度，再谈 group 间负载均衡

grouped matmul 的优化顺序很重要。

推荐顺序是:

1. 先保证单 group 内的 tile traversal 是对的
2. 再看多个 group 叠加后的 core 分配
3. 最后才考虑不规则 group_list 或 routed grouped matmul

原因很简单:

- 单 group 内的 tile 调度决定 Cube utilization
- group 间分发只是在这个基础上做更大的工作分桶

## 2.1 为什么不要一上来先优化 group_list

真实业务里的 grouped matmul 往往会带 `group_list` 或 expert/routing 信息，但这不是第一阶段最该优化的部分。

因为对规则 grouped matmul 来说，最大的性能来源仍然是:

- `base_m/base_n/base_k` 的选择
- `matmul` 和 `matmul_acc` 的累计结构
- `GM -> MAT -> L0 -> ACC -> GM` 的流水是否顺

如果这一层没站稳，过早把注意力放在 `group_list`:

- 只会把代码复杂度拉高
- 但并不能显著提升 Cube 密度

## 2.2 一个更稳的 persistent-kernel 写法

在 A3/910B 这类场景下，更稳的写法通常仍然是 persistent kernel:

```python
bid = pto.index_cast(pto.get_block_idx())
num_blocks = pto.index_cast(pto.get_block_num())

for logical_block in range(bid, total_group_tiles, num_blocks):
    ...
```

而不是让 launch grid 跟随 `G * m_tiles * n_tiles` 动态膨胀。

这样写的好处:

- 和 plain matmul 的 kernel launch 行为一致
- 更适合 PTO 的 autosync 和现有工程经验
- 不容易踩到 block_dim 过大导致的同步 / 调度问题

## 2.3 group 间负载不均匀怎么办

如果每个 group 的 shape 不同，或者某些 group 的有效 tile 数显著更少，那么可以考虑:

- 先把每个 group 的 tile 总数做 prefix sum
- 再把 `logical_block` 映射到 `(g_idx, local_tile)`

但这属于第二阶段优化。第一阶段先把规则 grouped matmul 跑顺更重要。

换句话说:

- **规则 grouped matmul**: 先用 `tiles_per_group` 常量映射
- **不规则 grouped matmul / routed GMM**: 再引入 `group_list`、prefix sum、routing map

# 第 3 步: 复用 swizzle，而不是重写一套新 schedule

本地 matmul guide 的一个重要结论是:

- swizzle 改的是 tile traversal
- 不改的是数学语义

这一点在 grouped matmul 上同样成立。

当前 grouped matmul seed 直接复用了 plain matmul 的两个 helper:

```python
from pto_kernels.ops.gmm.common import swizzle_nz, swizzle_zn
```

也就是说 grouped 的推荐顺序不是:

- 先为 grouped 发明一套新的 fancy schedule

而是:

- 先把 `local_tile` 视为“单 group 内的 tile 索引”
- 再直接调用已经验证过的 `swizzle_zn(...)` 或 `swizzle_nz(...)`

典型写法就是:

```python
m_idx = local_tile // n_tiles
n_idx = local_tile % n_tiles
if swizzle_direction == 0:
    m_idx, n_idx = swizzle_zn(local_tile, m_tiles, n_tiles, swizzle_count)
elif swizzle_direction == 1:
    m_idx, n_idx = swizzle_nz(local_tile, m_tiles, n_tiles, swizzle_count)
```

这里最值得强调的工程原则是:

- **group 维负责“选哪一组”**
- **swizzle 负责“这一组内部 tile 怎么走”**

不要把这两层逻辑搅在一起。

如果直接对全局 `logical_block` 做 swizzle，而不是对 group 内的 `local_tile` 做 swizzle，通常会导致:

- 同一个 group 的 tile 被分散得更厉害
- B 矩阵的 group 局部性下降
- 对后续 epilogue 融合更不友好

# 第 4 步: 从 grouped matmul 扩展到 grouped matmul add

一旦 grouped matmul 本体稳定，最自然的下一步是融合轻量 epilogue，例如:

```text
Y = grouped_matmul(X, W) + residual
```

当前迁移中的 `grouped_matmul_add` 采用的是非常务实的两阶段结构:

1. `stage_matmul`
   - `cube` section
   - 先完成 grouped matmul
   - 输出临时 `mm`
2. `stage_add`
   - `vector` section
   - 按行加载 `mm_row` 和 `y_row`
   - `pto.add(...)`
   - 写回 `out_row`

## 4.1 为什么先接受“两阶段”

很多人看到这里会问:

- 为什么不一开始就把 add 完全 fusion 进 cube kernel?

原因是对于教程和迁移初期来说，两阶段更稳:

- matmul 主体性能问题和 add epilogue 问题能分开看
- cube / vector 的职责更清晰
- correctness 更容易对齐

只有在确认 epilogue 成本已经显著成为瓶颈时，才值得继续做更深的 fusion。

## 4.2 两阶段 grouped_matmul_add 的结构

`stage_matmul` 与 grouped matmul 本体几乎一致:

```python
pto.load(sv_a, a_mat)
pto.load(sv_b, b_mat)
pto.mov(a_mat, a_tile)
pto.mov(b_mat, b_tile)
if i == c0:
    pto.matmul(a_tile, b_tile, c_tile)
else:
    pto.matmul_acc(c_tile, a_tile, b_tile, c_tile)
pto.store(c_tile, sv_out)
```

`stage_add` 则是典型的按行 vector epilogue:

```python
for row_idx in range(row_start, row_end, c1):
    pto.load(sv_mm, mm_row)
    pto.load(sv_y, y_row)
    pto.add(mm_row, y_row, out_row)
    pto.store(out_row, sv_out)
```

这个结构有两个优点:

- grouped matmul 主体仍然保留最清晰的 cube pipeline
- add 部分完全落在 vector path，上手简单，调试成本低

## 4.3 什么时候值得继续 fusion

如果下面两个条件同时出现，就该考虑进一步 fusion:

- `mm -> out` 的额外 GM round-trip 已经成为主要开销
- epilogue 不只是 add，而是更长的链条，例如 add + norm + activation

否则，两阶段往往是很好的工程平衡点。

# 第 5 步: 真正值得继续优化的方向

当 grouped matmul baseline 已经稳定后，真正有价值的优化方向通常是这几类。

## 5.1 Double buffering / preload

grouped 不会改变 matmul 的根本瓶颈，所以 plain matmul guide 里的下一步优化在这里仍然成立:

- A/B tile 双缓冲
- 预取下一 K tile
- 降低 Cube 等待数据的空泡

grouped 只是在更外层多了一层 group 选择，不会改变这条原则。

## 5.2 group-aware scheduling

如果 group 数量很多，且每组都很小，那么可以考虑:

- 让一个 core 连续处理同一个 group 的多个 tile
- 避免 group 间频繁切换

这通常比“所有 group 完全打散”更有利于局部性。

## 5.3 routed / quantized grouped GMM

真正复杂的 grouped GMM 通常不是普通 dense grouped matmul，而是:

- routed grouped matmul
- grouped matmul + swiglu
- quantized grouped matmul

这类问题里，新增复杂度大多来自:

- group/routing 元数据
- 权重格式
- quant/dequant / epilogue

但底层 cube matmul tile 方案依然应该尽可能继承 plain matmul 或 dense grouped matmul 的成熟路径。

也就是说:

- **不要为 routing/quant 过早牺牲 matmul 主路径**
- 先让 cube path 保持规整
- 再在外围补 group/routing/epilogue 逻辑

# 附录 A: 一个最小 PTO-DSL grouped matmul 骨架

下面这个骨架故意保留“和 plain matmul 非常像”的结构:

```python
@jit(...)
def grouped_matmul(out_ptr, a_ptr, b_ptr, group_count_i32):
    with pto.section.cube():
        bid = pto.index_cast(pto.get_block_idx())
        num_blocks = pto.index_cast(pto.get_block_num())

        tiles_per_group = cMTiles * cNTiles
        total_group_tiles = group_count_i32 * tiles_per_group

        tv_a = pto.as_tensor(tensor_a, ptr=a_ptr, shape=[cG, cM, cK], strides=[cM * cK, cK, c1])
        tv_b = pto.as_tensor(tensor_b, ptr=b_ptr, shape=[cG, cK, cN], strides=[cK * cN, cN, c1])
        tv_out = pto.as_tensor(tensor_out, ptr=out_ptr, shape=[cG, cM, cN], strides=[cM * cN, cN, c1])

        a_mat = pto.alloc_tile(tile_a_mat)
        b_mat = pto.alloc_tile(tile_b_mat)
        a_tile = pto.alloc_tile(tile_a)
        b_tile = pto.alloc_tile(tile_b)
        c_tile = pto.alloc_tile(tile_c)

        for logical_block in range(bid, total_group_tiles, num_blocks):
            g_idx = logical_block // tiles_per_group
            local_tile = logical_block % tiles_per_group
            m_idx = local_tile // cNTiles
            n_idx = local_tile % cNTiles

            m_idx, n_idx = swizzle_nz(local_tile, cMTiles, cNTiles, cSwizzleCount)

            m_off = m_idx * cTileM
            n_off = n_idx * cTileN

            for k_idx in range(c0, cKIter, c1):
                k_off = k_idx * cTileK
                sv_a = pto.slice_view(view_a, source=tv_a, offsets=[g_idx, m_off, k_off], sizes=[c1, cTileM, cTileK])
                sv_b = pto.slice_view(view_b, source=tv_b, offsets=[g_idx, k_off, n_off], sizes=[c1, cTileK, cTileN])
                pto.load(sv_a, a_mat)
                pto.load(sv_b, b_mat)
                pto.mov(a_mat, a_tile)
                pto.mov(b_mat, b_tile)
                if k_idx == c0:
                    pto.matmul(a_tile, b_tile, c_tile)
                else:
                    pto.matmul_acc(c_tile, a_tile, b_tile, c_tile)

            sv_out = pto.slice_view(view_out, source=tv_out, offsets=[g_idx, m_off, n_off], sizes=[c1, cTileM, cTileN])
            pto.store(c_tile, sv_out)
```

如果读完这个骨架你觉得“除了多了个 `g_idx`，几乎还是 plain matmul”，那说明方向是对的。

# 附录 B: grouped matmul 调参 checklist

真正开始调 grouped matmul 时，建议按这个顺序检查:

1. `base_m/base_n/base_k` 是否和 plain matmul 的硬件约束一致
2. `group 维` 是否只是顶层工作分发，而不是侵入到 hot K loop
3. swizzle 是否只作用于 group 内的 `local_tile`
4. `block_dim` 是否仍按 persistent-kernel 风格选择
5. 是否已经区分:
   - dense grouped matmul
   - grouped matmul add
   - routed / quantized grouped GMM
6. 如果性能不佳，先查:
   - core 利用率
   - GM/L1/L0 数据流
   - 是否有多余的 group 间切换
7. 不要在还没把 dense grouped matmul 跑顺之前，就过早引入复杂 routing / quant 元数据

最后给一句经验判断:

- **plain matmul 优化得好的 kernel，通常已经有 70% 的 grouped matmul 答案**
- 剩下的 30% 才是 group-aware scheduling 和 epilogue / routing / quant 的工程细节
