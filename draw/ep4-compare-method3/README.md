# Figure 6 说明：EP4 跨层专家重排（Cross-Layer Expert Rearrangement）对比图

论文当前版本（`iclr2027` 分支）里的 Figure 6，caption 是：

> Comparison of EP4 (4-way expert-parallel) GPU rearrangement strategies for
> the same width-heterogeneous pruned MoE. (a) A naive static rearrangement
> always sends the same width tier to the same GPU in every layer, causing
> large, persistent memory imbalance. (b) A width-agnostic baseline assigns
> experts to GPUs by contiguous ID blocks, ignoring width; each GPU ends up
> mixing tiers and needs multiple fused-MoE calls per layer. (c) Our
> cross-layer greedy rearrangement keeps exactly one tier per GPU per layer
> but rotates which tier goes where across layers, balancing cumulative
> per-GPU memory.

**这张图目前是占位图（合成数据），不是真实实验结果。** 本 README 用中文把图的含义、
构造方式、以及后续要替换成真实数据时需要做什么，写清楚，供你跑出真实数值之后对照。

绘图脚本的当前位置（论文仓库那边，不在这个代码仓库里）：

```text
2026-NIPS-Multimodal-MoE/draw/ep4-compare-method3/plot_ep4_placement_compare.py
```

## 1. 这张图想说明什么问题

论文的方法三（4.3 节，"Greedy-Based Cross-Layer Expert Rearrangement for
Multi-GPU Parallelism"）要解决的问题是：channel-level 剪枝之后，每个 expert
保留的通道宽度是不一样的（宽度异构）。要在 4 卡 expert-parallel（EP4）上部署，
fused MoE kernel 要求"同一层、同一张 GPU 上的所有 local expert 宽度必须一致"，
所以要先把连续宽度量化到几个硬件友好的档位（tier），再决定"每层的这几个宽度档
分别放到哪张卡上"——这就是"跨层重排（cross-layer rearrangement）"问题。

Figure 6 用三个子图对比三种"档位放到哪张卡"的策略，验证的核心论点是：**朴素的
放置方式会导致长期、结构性的显存不均衡，而本文的跨层贪心重排能把这个不均衡降到
几乎为零**。三个子图用的是**完全相同的逐层宽度量化结果**（哪个 expert 该给多宽，
三个子图都一样），唯一的区别是"放到哪张卡"这个决策本身，这样才是公平对比放置
策略，而不是在比不同的量化结果。

- **(a) Static per-layer placement (naive)**：每一层都固定用同一种"档位→GPU"的
  对应关系（比如永远是"384 档给 GPU0，512 档给 GPU1，640 档给 GPU2，768 档给
  GPU3"），层与层之间从不换。因为不同层里各档位的 expert 数量本来就不均匀，
  固定映射会让某几张卡长期偏重（图里 GPU1、GPU2 明显比 GPU0、GPU3 重）。
- **(b) ID round-robin (width-agnostic baseline)**：完全不看宽度，只按 expert
  的编号，每 16 个一组，按顺序轮流分给 4 张卡（0-15 号给 GPU0，16-31 号给
  GPU1，以此类推，循环两轮凑满 128 个 expert）。因为一张卡分到的 16 个 expert
  里可能同时混有好几种宽度档位，所以**同一张卡在同一层可能需要调用好几次
  fused-MoE kernel**（每种宽度一次），这是它区别于 (a) 和 (c) 的关键缺点——不是
  显存问题，是要多次 kernel 调用的效率问题。
- **(c) Cross-layer greedy rearrangement (ours)**：本文方法。每一层仍然是"一张卡
  一个档位"（不会像 (b) 那样在同一张卡上混宽度，所以不需要多次 kernel 调用），
  但**每一层的"档位→GPU"映射可以不一样**，用贪心构造 + 局部搜索来决定每一层具体
  怎么映射，使得把 48 层全部放完之后，四张卡累计下来的显存尽量接近。

一句话总结三者的取舍：(a) 简单但长期不均衡；(b) 不均衡问题较小但需要多次 kernel
调用；(c) 既不需要多次 kernel 调用，又能做到几乎完全均衡。

## 2. 配色约定（最近改过一版）

**旧版本**：颜色按"宽度档位"编码（4 种颜色对应 384/512/640/768 四档），每张卡的
横向柱子是 48 层的小段拼起来的"花纹"，用来展示"某一层的哪个档位被放到了哪张卡"。

**当前版本**：改成了按"策略/子图"编码，和论文 Figure 1(b)/(c) 里 Padding /
MultiKernel / Ours 三种部署策略用的是同一套颜色：

| 子图 | 策略 | 颜色 | 对应 Figure 1(b)/(c) 里的角色 |
|---|---|---|---|
| (a) | Static per-layer placement (naive) | 浅紫色 `#C9B3E0` | 类比 Padding——最朴素、最容易想到的做法 |
| (b) | ID round-robin (width-agnostic baseline) | 浅灰色 `#B5B5B5` | 类比 MultiKernel——都需要"同一层多次调用 fused-MoE kernel"这个缺点 |
| (c) | Cross-layer greedy rearrangement (ours) | 橙色 `#E66100` | 和全文所有"Ours"保持同一个颜色 |

这样改的原因：一是让全文的颜色语言统一（"橙色=我们的方法"这件事在所有图里都一样），
二是因为这张图接下来要换成真实实验数据，真实数据大概率就是"每张卡的总显存"这种
单一数字，天然适合用单色柱子表示，不再需要展示"哪一层的哪个档位落在哪张卡"这种
逐层细节。

## 3. 目前用的是什么合成数据（构造方式）

绘图脚本里的数据是根据 `ep4_intplan.py`（真实代码库 `Multimodal-MoE-Slimming`
的 `ep4_intplan` 分支）里描述的算法结构模拟出来的，规模上对齐 Qwen3-VL-30B-A3B：

- `L=48` 层，`E=128` 个 expert/层，4 个非零宽度档 `{384, 512, 640, 768}`。
- 每层的档位构成 `n[l,k]`（第 `l` 层落在第 `k` 档的 expert 数）：先随机生成一个
  "层敏感度"，归一化后决定这一层的目标平均宽度，再用高斯权重把 128 个 expert
  的名额撒到 4 个档位上，同时强制每层每个档位至少有 1 个 expert（模拟真实算法里
  "匈牙利算法精确锚定四个强制档位"这一步的效果，但这里不是真的在跑匈牙利算法，
  只是保证这条硬约束成立）。
- (b) 子图额外需要"每层具体是哪几个 expert 落在哪个档位"（不只是数量），这部分
  用了一个固定的、跨层持续存在的"ID 倾向性"函数（几个局部的高斯"热点"，不是
  周期函数，避免 16 个一组的轮询区间正好把周期性抵消掉），保证按 ID 分块之后
  真的会产生持续性的、不是偶然的不均衡。
- 放置算法本身是真的：(a) 固定 identity 排列；(c) 是贪心构造（按层内负载极差
  降序决定顺序，每层选当前最优排列）加坐标下降局部搜索（反复对每层做 1-opt
  调整直到收敛），和论文里描述的算法完全一致，不是示意性的假算法。

**所以这张图里，"策略"和"算法"是真的，"每层每个 expert 具体多宽"是编出来的。**

## 4. 换成真实数据时需要准备什么

1. **真实的逐层档位构成 `n[l,k]`**：从 `ep4_intplan.py` 的
   `_quantize_widths_to_budget`（对应 `plan_ep4_intplan` 里的 `width_counts`
   返回值）里直接读出来，不需要重新模拟。需要跑一次真实的 Qwen3-VL-30B-A3B
   在某个具体剪枝率下的 tier quantization。
2. **(b) 子图需要每个 expert 具体分到了哪个档位**（不只是每档数量），这部分
   `_quantize_widths_to_budget` 应该已经直接输出了（`expert_widths` 或类似字段），
   不需要再用合成的"ID 倾向性"去猜。
3. **真实的 GPU 显存换算**：目前脚本里 `mem_gb()` 用的是
   `3 * d_model * m_e * bytes_per_param` 这个理论参数量公式（`d_model=2048` 是
   占位数值），换成真实数据时，应该直接用 checkpoint 的真实 `d_model`，或者更好
   的做法是直接测量真实显存占用（类似 `draw/intro-3panel` 那张图里
   `max_non_kv_peak_gib` 的做法），而不是用理论参数量估算。
4. 如果要保留"多次 fused-MoE kernel 调用"这个 (b) 方案的缺点在图上可视化出来
   （不只是文字描述），可以像旧版本一样在 (b) 子图里把同一张卡的柱子按"落在
   这张卡的不同宽度档位"分段着色，或者单独加一个"每层每卡的 kernel 调用次数"
   的统计侧栏——这个目前没做，纯粹是设计取舍，不是技术限制。

## 5. 复现当前占位图

在论文仓库（不是这个代码仓库）里运行：

```bash
python3 draw/ep4-compare-method3/plot_ep4_placement_compare.py
```

会在同目录下生成 `ep4_placement_compare.pdf` / `.png`，并打印三种策略每张卡的
显存数值和最大偏差百分比。论文正文引用的图片是
`Maes_NeurIPS_2026/imgs/ep4-compare.pdf`，需要手动把生成的 PDF 拷贝过去。
