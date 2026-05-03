# Modality-aware Channel Budget 分配方法

## 1. 问题背景

在 modality-aware 剪枝里, 每个 expert 同时有两套 channel 重要性排序:

- `visual_scores`, 表示视觉模态视角下的 channel 重要性.
- `text_scores`, 表示文本模态视角下的 channel 重要性.

这两套分数对应的并不是两组独立 channel, 而是同一组 channel 的两种不同排序. 因此, 如果直接分别取 `topk_visual` 和 `topk_text`, 再对两者做并集, 会出现大量重复 channel. 结果就是:

1. 虽然看起来已经给视觉和文本分别分配了预算.
2. 但最终 unique channel 的数量可能小于目标 budget.
3. 这会导致每个 expert 的实际保留通道数不稳定.

此外, 当我们引入 EMA affinity 去调节 visual/text 的预算比例时, 如果对每个 expert 独立使用

\[
\text{target}_e = w_e^{text} \cdot K_e^{text} + w_e^{vis} \cdot K_e^{vis}
\]

则每个 expert 的 target 会发生偏移. 这本身没有问题, 但所有 expert 的 target 相加后, 往往不再等于该层原始的 layerwise budget. 这样会破坏整层的通道总量约束.

因此, 当前实现同时解决两个问题:

- 在单个 expert 内, 避免 visual/text 重复选中同一 channel 后造成预算不足.
- 在整层范围内, 保证所有 expert 的最终 channel 数总和严格等于原始 layerwise budget.

## 2. 整体思路

整个过程分为两个层次:

1. expert 内部的 unique channel 选择.
2. layer 内部的预算守恒分配.

具体来说:

- 首先, 仍然分别基于 `visual_scores` 和 `text_scores` 计算 tentative mask, 得到每个 expert 在两种模态视角下的初始预算 `K_e^{vis}` 和 `K_e^{text}`.
- 然后, 根据 EMA affinity 计算 visual/text 的权重.
- 接着, 先在 layer 级别固定这一层的总预算, 再把总预算重新分配给各个 expert.
- 最后, 在每个 expert 内部, 按照 visual/text 的权重比例, 用二分搜索驱动的逐步扩张算法选择 unique channel.

## 3. EMA 权重定义

对第 `l` 层第 `e` 个 expert, 记其 affinity 为 `a_{l,e}`. 其中:

\[
a_{l,e} \in [-1, 1]
\]

将其映射到 visual 权重:

\[
w_{l,e}^{vis} = \frac{a_{l,e} + 1}{2}
\]

文本权重定义为:

\[
w_{l,e}^{text} = 1 - w_{l,e}^{vis}
\]

当不使用 EMA 时, 直接令:

\[
w_{l,e}^{vis} = w_{l,e}^{text} = 0.5
\]

这样, 模型会在视觉优先, 文本优先, 或均匀分配三种情形之间自然切换.

## 4. 单个 expert 内的 unique channel 选择

### 4.1 直接取 top-k 的问题

如果直接执行:

- 从 `visual_scores` 中取前 `k_vis` 个 channel.
- 从 `text_scores` 中取前 `k_text` 个 channel.
- 最终保留集合为两者的并集.

由于两边可能反复命中同一个 channel, 实际 unique channel 数经常小于 `target_budget`.

例如:

- visual 排序前几个 channel 为 `[1, 3, 5, 7, 9]`.
- text 排序前几个 channel 也可能是 `[1, 5, 2, 7, 4]`.

若分别取 3 个, 得到的并集可能只有 `{1, 3, 5, 2}` 或 `{1, 3, 5}` 这样的结果, 小于原本期望的 6 或 5 个 unique channel.

### 4.2 逐步扩张的思路

为了解决这个问题, 当前实现不再一次性计算最终 `topk_visual` 和 `topk_text`, 而是使用一个"逐步扩张排序前缀"的过程:

- visual 和 text 各自维护一个按分数从高到低的有序列表.
- 算法每次只从其中一侧取"下一个" channel.
- 选择哪一侧, 由当前已消费的 visual/text 步数是否匹配目标权重来决定.

也就是说, 如果某个 expert 的权重为:

\[
w^{vis} = 0.8, \quad w^{text} = 0.2
\]

那么在长程统计上, 算法会尽量逼近 `4:1` 的扩张节奏. 这不意味着每连续 5 步一定严格是 4 个 visual 加 1 个 text, 而是指整个扩张路径会持续朝这个比例靠拢.

### 4.3 重复 channel 的处理

由于 visual 和 text 来自同一组 channel, 某一步新加入的 channel 可能已经在当前集合中出现过. 当前实现对这种情况的处理是:

- 该 channel 虽然不会增加 unique 集合大小.
- 但仍然记作该模态消费了一次配额, 并继续向后推进该模态的排序指针.

这样做的原因是:

1. 它保持了"按排序逐步扩张"这一语义.
2. 它不会因为重复命中而反复卡在同一个位置.
3. 它使 visual/text 两侧的扩张过程仍然服从预期比例.

### 4.4 二分搜索最小扩张步数

设从 visual/text 两边总共扩张 `s` 步后, 最终得到的 unique channel 数为 `U(s)`. 可以观察到:

- 随着 `s` 增大, `U(s)` 单调不减.
- 因此可以对 `s` 做二分搜索.

算法目标是找到最小的 `s^*`, 使得:

\[
U(s^*) \ge \text{target\_budget}
\]

这样做有两个好处:

1. 最终得到的 unique channel 数能够达到目标预算.
2. 扩张步数尽可能小, 也就是尽量保留更高分的 channel, 避免无谓地向排序尾部扩张.

## 5. shared protect 与 no shared protect 的统一处理

当前实现中, `shared_protect=True` 和 `shared_protect=False` 都使用同一套二分搜索扩张逻辑, 区别只在候选集合的定义方式.

### 5.1 `shared_protect=True`

先将 tentative mask 分解为三部分:

- `shared_mask`, 同时被 text 和 visual 选中的 channel.
- `text_only_mask`, 只被 text 选中的 channel.
- `visual_only_mask`, 只被 visual 选中的 channel.

其中:

- `shared_mask` 中的 channel 会被直接保留.
- 它们构成当前 expert 的初始已选集合 `chosen_base`.
- 之后只在 `text_only_mask` 与 `visual_only_mask` 中继续做按比例扩张和二分搜索.

因此, shared protect 的本质是把共享高价值 channel 作为硬保留集合, 剩余预算再通过排序扩张补齐.

### 5.2 `shared_protect=False`

不再预先固定共享集合, 而是直接在完整 channel 集合上, 同时利用 visual 排序和 text 排序做扩张.

虽然这一分支没有显式的 shared mask, 但仍然面临 visual/text 排序重叠的问题, 因此同样必须使用 unique-aware 的二分搜索选择算法.

## 6. 层级预算守恒

### 6.1 为什么需要层级守恒

如果对每个 expert 独立计算:

\[
\text{raw\_target}_{l,e} = w_{l,e}^{text} \cdot K_{l,e}^{text} + w_{l,e}^{vis} \cdot K_{l,e}^{vis}
\]

则这些 `raw_target` 体现了 EMA 对不同 expert 的偏置. 但是:

\[
\sum_e \text{raw\_target}_{l,e}
\]

一般不再等于该层原始的 layerwise budget. 于是整层保留通道数会漂移, 这会破坏上游由 `layerwise_keep_plan` 规定的剪枝率约束.

### 6.2 层目标预算的确定

对第 `l` 层, 设:

- expert 数为 `E`.
- 每个 expert 的 channel 数为 `I`.
- 原始 layerwise keep ratio 为 `r_l`.

则该层的目标总 budget 为:

\[
B_l = \lceil r_l \cdot E \cdot I \rceil
\]

实现里还对浮点误差做了轻微修正, 避免如 `0.6 * 10` 因数值误差被错误上取整到 7.

如果 `layerwise_keep_plan` 本身已经是 `[L, E]` 形式, 则先将每个 expert 的 keep ratio 转为对应预算, 再在层内求和得到 `B_l`.

### 6.3 expert 预算再分配

在确定层总预算 `B_l` 后, 需要把它重新分配给各个 expert. 当前实现对每个 expert 定义:

- `raw_target_{l,e}`, 表示 EMA 加权后的软目标.
- `min_budget_{l,e}`, 表示该 expert 至少必须保留的通道数.
- `max_budget_{l,e}`, 表示该 expert 最多可保留的通道数.

其中:

- 在 `shared_protect=True` 时, `min_budget` 至少等于 shared mask 的大小, 因为这些共享 channel 必须保留.
- 在 `shared_protect=False` 时, `min_budget` 通常为 0.
- `max_budget` 则由该 expert 可选 channel 总数决定.

随后, 算法在约束

\[
\sum_e B_{l,e} = B_l
\]

的前提下, 按照 `raw_target` 的相对大小为各个 expert 分配 budget `B_{l,e}`.

当前分配方式是:

1. 先为所有 expert 分配 `min_budget`.
2. 剩余预算按照 `raw_target - min_budget` 的相对权重进行分配.
3. 先取 floor 部分.
4. 如果还存在剩余预算, 再按小数部分从大到小补齐.
5. 同时始终满足 `B_{l,e} \le max_budget_{l,e}`.

这样得到的 `B_{l,e}` 具有两个性质:

- 整层总预算严格守恒.
- expert 间预算分布仍然尽量贴近 EMA 加权得到的软目标.

## 7. 完整算法流程

对每一层 `l`, 当前实现可概括为如下步骤:

1. 基于 `text_scores` 生成 text tentative mask, 得到 `K_{l,e}^{text}`.
2. 基于 `visual_scores` 生成 visual tentative mask, 得到 `K_{l,e}^{vis}`.
3. 对每个 expert 计算 EMA 权重 `w_{l,e}^{vis}` 与 `w_{l,e}^{text}`.
4. 对每个 expert 计算 `raw_target_{l,e}`.
5. 根据 `layerwise_keep_plan` 计算整层总预算 `B_l`.
6. 结合 `raw_target`, `min_budget`, `max_budget`, 将 `B_l` 守恒地分配为各个 expert 的 `B_{l,e}`.
7. 对每个 expert, 使用按比例逐步扩张 + 二分搜索, 从 visual/text 排序中选择 unique channel, 直到达到 `B_{l,e}`.
8. 汇总得到最终 mask.

## 8. 方法特点

这套方法相对于简单的 top-k 并集策略, 主要有以下优点:

- 能显式处理 visual/text 两套排序对应同一组 channel 的重叠问题.
- 能在重复 channel 较多时仍然稳定达到目标 unique budget.
- 能在使用 EMA affinity 时保留模态偏置.
- 能同时严格满足 layerwise 总预算约束.
- `shared_protect=True` 与 `False` 两个分支共享统一算法框架, 行为更一致.

## 9. 适合论文中的表述重点

如果你要把这一段写进论文方法部分, 建议重点强调以下三点:

1. 我们不是独立地从 visual/text 中各取一批 channel, 而是把它们视为同一组 channel 的两种排序, 并以 unique set 为优化对象.
2. 我们通过按比例逐步扩张和二分搜索, 在满足模态权重偏好的同时, 精确控制最终 unique channel 数.
3. 我们在 layer 级别施加预算守恒约束, 从而保证 modality-aware 分配不会破坏全局剪枝率计划.
