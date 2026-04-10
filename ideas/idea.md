ide | ToMe, Token Pruning | 减少输入 token 数量 |
| Execution-side | Expert skipping, Sparse routing | 减少激活的 expert 数量 |
| Capacity-side（本文） | 通道级模态感知剪枝（本文） | 压缩 expert 内部权重 |

**核心主张**：multimodal MoE 中存在 *模态特异性通道冗余*——即某个 expert 内部的部分通道高度响应某种模态，而 router 在实际推理中并不会把该模态的 token 分配给这个 expert。这类"无效容量"可以被安全裁掉，与 token-side 和 execution-side 正交，形成第三压缩维度。

---

### **观察实验**

**实验 O1：Expert 模态偏好分析（Router 视角）**

**目的**：证明不同 expert 对不同模态 token 有显著的路由偏好。

**方法**：

1. 收集多模态校准集（文本 token 集合 \mathcal{D}_T，图像 token 集合 \mathcal{D}_V），各 ~1K 样本。
2. 对每一层 l、每个 expert e，统计其被不同模态 token 激活的频率：

$\text{RoutingFreq}(e, l, m) = \frac{1}{|\mathcal{D}_m|} \sum_{x \in \mathcal{D}_m} \mathbf{1}[\text{router assigns } x \text{ to } e]$

1. 定义 **Expert Modality Affinity (EMA)**：

$\text{EMA}(e, l) = \frac{\text{RoutingFreq}(e, l, V) - \text{RoutingFreq}(e, l, T)}{\text{RoutingFreq}(e, l, V) + \text{RoutingFreq}(e, l, T)}$

$\text{EMA} \in [-1, 1]$，+1 表示纯视觉 expert，-1 表示纯文本 expert，0 表示中性。

**预期发现**：EMA 在不同 expert 间分布不均，存在明显模态偏好（部分 expert 有强 $|\text{EMA}| > 0.5$）。

**可视化**：对所有层的 EMA 画热力图（x: expert id, y: layer id），观察模态偏好是否有层级规律（如早期层模态中性、深层出现分化）。

**实验 O2：Expert 内通道模态响应分析（Channel 视角）**

> 
> 
> 
> 看是否有这个结论作为 motivation 出发点：
> 
> - 不同模态对 expert 内部通道使用并不均衡。MoE  的不同expert、不同通道，对不同模态 token 的贡献存在显著差异
> - ⭐️ 某些通道在某个模态 token 下激活显著，但是在实际 router 使用时，并不会将这个模态 token 分配给这个专家，就可以将这个通道剪枝

**目的**：核心动机验证——expert 内部通道对不同模态响应是否存在显著差异。

**方法**：

1. 对于每个 expert e 的中间激活维度（FFN up-proj 后的通道），定义通道 c 对模态 m 的激活强度：

$\text{ChanResp}(c, e, m) = \mathbb{E}_{x \sim \mathcal{D}_m} \left[ |a_c^e(x)| \right]$

其中 $a_c^e(x)$ 为 token x 被路由到 expert e 时，通道 c 的激活值（注意：只统计**实际被路由到该 expert** 的 token）。

> ⚠️ 关于激活 vs. 激活+梯度的选择：
> 
> - **纯激活** $|a_c|$：反映通道的表征强度（快速计算，适合 pilot）
> - **激活 × 梯度** $|a_c| \cdot |\nabla_{a_c} \mathcal{L}|$：反映通道对损失的敏感度（MAS 完整定义）
> - **推荐**：pilot 阶段用纯激活验证存在性，正式方法用激活+梯度提升精度
1. 定义通道的模态偏向性：

$\text{ModalBias}(c, e) = \frac{|\text{ChanResp}(c, e, V) - \text{ChanResp}(c, e, T)|}{\text{ChanResp}(c, e, V) + \text{ChanResp}(c, e, T) + \epsilon}$

**预期发现**：即使在文本偏向 expert（EMA < 0）内部，仍存在部分高 ModalBias 的视觉响应通道（反之亦然）。

**可视化**：对于一个选定的层，画出各 expert 内的通道激活分布（按模态分类），用双色柱状图对比。

**实验 O3：⭐ Conflict 验证实验（动机核心）**

**目的**：验证最关键的 motivation——"router 不选择该 expert，但该 expert 内有对该模态响应的通道"这一 conflict 现象的普遍性。

**方法**：

对于每个 expert e 和模态 m：

- **Router 偏好**：$\text{RoutingFreq}(e, l, m)$ 低（说明 router 不把模态 m 分给 e）
- **Channel 响应**：$\text{ChanResp}(c, e, m)$ 高（说明 e 内有对模态 m 响应的通道）

定义 **Conflict Score**：

$\text{ConflictScore}(e, l, m) = \underbrace{(1 - \text{NormRoutingFreq}(e, l, m))}_{\text{router 回避程度}} \times \underbrace{\text{MeanModalBias}(e, m)}_{\text{内部通道强响应程度}}$

统计 ConflictScore 高的 $(e, l, m)$ 三元组占比，形成"无效容量图"。

**预期发现**：存在系统性的 high-conflict expert（router 极少路由该模态，但内部通道对该模态有强响应），支撑可剪枝性。

---

### **初步方法**

**`Idea 1 Modality Affinity`**

> 
> 
> 
> **expert modality affinity matrix 模态亲和度矩阵**
> 
> - 哪些 expert 偏视觉或偏文本，这个看 router output
> 
> todo: 如何设计模态亲和度计算方法？
> 
> **channel-wise modality response intensity 响应强度**
> 
> 对于 expert 中的通道 `c`，定义其对模态 `m` 的重要性
> 
> todo：响应强度如何定义和计算？激活 or 激活+梯度？激活可以看作响应强度，梯度更偏敏感度？
> 

**Step 1: 构建 Modality Affinity Score（MAS）**

$\text{MAS}(c, e, m) = \mathbb{E}_{x \sim \mathcal{D}_m^e} \left[ |a_c^e(x)| \cdot |\nabla_{a_c^e} \mathcal{L}(x)| \right]$

其中 $\mathcal{D}_m^e$ 表示**被实际路由到 expert e 的模态 m 的 token 集合**（关键：只用到达该 expert 的 token，而非全部模态 token）。

**Step 2: 确定剪枝目标通道**

对 expert e 内通道 c，综合其对所有被分配模态的重要性：

$\text{PruneScore}(c, e) = \max_m \text{MAS}(c, e, m) \times \text{RoutingFreq}(e, l, m)$

> 这里的 routing frequency 加权是关键：如果 router 几乎不把模态 m 分给 expert e，那么即使通道 c 对模态 m 有响应，其实际影响也极小，可以剪掉。
> 

剪枝目标：$\text{PruneScore}(c, e)$  最低的通道。

**Step 3: 层级自适应剪枝率**

不同层设置不同的剪枝率（基于 O4 实验的层级冗余分布）

**`计算一个 conflict rule`**

- 针对某一模态 token，router 偏好的专家 vs. 偏好专家内实际对这个 modality 响应的 channel
- conflict 体现在：router 更喜欢把模态 A 分配给专家 e，而专家 e 中包含对模态A响应的 channel 也包含对模态 A 不响应的 channel，不响应的 channel 可以 pruned（不是 skipped 也不是 masked，就是直接剪枝，缩小权重存储量）

**`Layer-wise Adaptation`**

不同 depth 对不同模态的剪枝率不同，这个创新点比较小，只能作为补充，不作为单个 idea。

**`Idea 2 跨层合并冗余 expert`**

从"单层内的 expert 优化"扩展到"跨层的 expert 重组”，创新性较大

合并策略的优化空间很大，搜索成本高，方法容易复杂？比较偏向 expert merge 类的思路，但比较有意思。也可以做。

目前已知的 expert 合并类工作均集中在同层内，真正的跨层合并（将来自不同层的 expert 合并为一个 shared expert）目前尚无同类研究。

| **工作** | **合并方式** | **是否跨层** | **与 Idea2 的差异** |
| --- | --- | --- | --- |
| **EEP (Zhang et al., 2025, arXiv 2407.09590)** | CKA 衡量同层 expert 相似度，图聚类后 weight-averaging 合并，整体删除多余 expert | ❌ 同层内 | 无模态感知，无跨层 |
| **HC-SMoE (Chen et al., 2024)** | 层内 hierarchical clustering，retraining-free 合并 | ❌ 同层内 | 无模态感知，无跨层 |
| **GRAPE (Zhang et al., 2026, arXiv 2604.06542)** | 跨层**分配剪枝预算**（每层剪多少由全局冗余决定），但剪枝操作仍在层内执行 | ⚠️ 跨层感知预算，但不跨层合并 expert | 无模态感知，不做真正的跨层 expert 合并 |
| **MoE-Pruner (Xie et al., 2024, arXiv 2410.12013)** | 利用 router hints 做权重剪枝（unstructured） | ❌ 同层内 | 无 expert 合并 |

---

方法设计注意点：与普通 magnitude 剪枝的差异。普通剪枝直接拿文本+视觉作为 calib data 是否能达到一样的结果？

与普通 Magnitude 剪枝的对比：

| **维度** | **普通 Magnitude 剪枝** | **Idea1 (MAS-based)** |
| --- | --- | --- |
| 校准数据 | 混合文本+视觉 token | 分模态、分 expert 的 token 子集 |
| 剪枝粒度 | 全局通道重要性 | Expert \times Modality 条件下的通道重要性 |
| 是否感知 routing | ❌ | ✅（用 routing frequency 加权） |
| 能否挖掘 conflict | ❌（混合数据掩盖模态差异） | ✅ |

