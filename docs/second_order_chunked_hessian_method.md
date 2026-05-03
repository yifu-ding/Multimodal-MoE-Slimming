# 分块 Hessian 对角的二阶 Expert 打分方法

## 1. 背景

在 expert 级别的重要性估计中, 一个常见思路是利用一阶与二阶信息近似评估"移除某个 expert 输出后会带来多大损失变化". 为此, 当前实现引入了基于 expert scaling variable 的二阶打分方式.

设某一层共有 `E` 个 expert. 对每个 expert `e`, 我们引入一个可微缩放变量 `\alpha_e`, 并将它作用在该 expert 的输出上. 然后围绕当前点 `\alpha = 1` 计算 loss 对 `\alpha` 的一阶与二阶导数.

在原始实现中, 为了获得所有 active experts 的 Hessian 对角项, 会一次性构造单位矩阵并调用 batched autograd, 直接求出完整的一批二阶导结果. 这种方法在 active expert 数较多时会占用较高显存.

因此, 当前实现改为分块计算 Hessian 对角, 在保持二阶打分定义不变的前提下显著降低峰值显存开销.

## 2. 基本打分形式

记 loss 关于 `\alpha_e` 的一阶导为:

\[
g_e = \frac{\partial \mathcal{L}}{\partial \alpha_e}
\]

二阶导对角项为:

\[
h_e = \frac{\partial^2 \mathcal{L}}{\partial \alpha_e^2}
\]

当前实现使用的二阶近似打分为:

\[
s_e = \max\left(0, -g_e + \frac{1}{2} h_e \right)
\]

这可以理解为围绕当前点对移除 expert 或减小其贡献时的局部损失变化做二阶近似.

如果二阶项不可得, 则退化为一阶近似:

\[
s_e = \max(0, -g_e)
\]

## 3. 为什么需要分块

在一次 forward-backward 后, 我们可以得到 active experts 上的一阶导向量 `g`. 为了进一步得到 Hessian 对角, 一种直接做法是:

1. 取 active experts 的一阶导 `g_active`.
2. 构造大小为 `A \times A` 的单位矩阵作为 `grad_outputs`, 其中 `A` 是 active expert 数.
3. 调用 `torch.autograd.grad(..., is_grads_batched=True)` 一次性求得所有二阶行向量.
4. 再从中提取对角线.

这种做法的主要问题是:

- 当 `A` 较大时, batched Hessian-vector product 的中间张量会变大.
- 峰值显存往往由这一部分主导.
- 在大模型, 大 batch, 或 active experts 较多时容易造成 OOM.

因此, 当前实现不再一次性计算全部 active experts 的 Hessian 行, 而是分块执行.

## 4. 分块 Hessian 对角计算流程

设 active experts 的索引集合为:

\[
\mathcal{A} = \{e_1, e_2, \dots, e_A\}
\]

设分块大小为 `C`, 由环境变量 `SECOND_ORDER_CHUNK_SIZE` 控制, 默认值为 8.

算法流程如下:

1. 先完成一次一阶梯度计算, 得到 `g_e`.
2. 将 active experts 划分为若干块, 每块最多包含 `C` 个 expert.
3. 对于每一块:
   - 取当前块对应的一阶导分量.
   - 构造该块大小的单位矩阵 `I_C`.
   - 调用 `torch.autograd.grad` 计算这一块对应的 Hessian 行.
   - 仅提取当前块相关的对角元素.
4. 将各块得到的对角项拼接回完整的 Hessian diagonal.
5. 最终计算二阶打分 `s_e`.

换句话说, 该方法只是在"如何拿到 Hessian 对角"这一实现层面做了分块, 并没有改变最终的二阶打分目标.

## 5. 为什么分块后结果不变

Hessian 对角项本质上是逐 expert 的二阶偏导:

\[
h_e = \frac{\partial^2 \mathcal{L}}{\partial \alpha_e^2}
\]

无论是一次性求出全部 active experts 的 Hessian 行, 还是分多次求若干子块, 只要最终提取的是同样的对角元素, 得到的 `h_e` 在数学定义上是一致的.

因此, 分块策略改变的是:

- 计算图在每次 `autograd.grad` 中处理的输出维度大小.
- 中间张量的峰值规模.

而不会改变最终二阶打分的定义.

## 6. 显存优势

分块策略的核心好处是控制峰值显存. 相较于一次性处理全部 `A` 个 active experts:

- 原始方法需要构造 `A \times A` 规模的 batched 梯度输出.
- 分块方法每次只需处理 `C \times C` 量级的局部块.

当 `C << A` 时, 显存峰值会显著下降. 这使得二阶打分可以在更大的 batch 或更多 active experts 下稳定运行.

当前实现还额外打印了 CUDA 显存统计信息, 包括:

- 当前 allocated memory.
- 当前 reserved memory.
- 历史 max allocated memory.

这些统计有助于实际 profiling 和 chunk size 调参.

## 7. fallback 机制

在某些情况下, 二阶信息可能无法稳定取得, 例如:

- 当前块的 Hessian 相关梯度返回 `None`.
- 某些 expert 在当前图中实际上没有形成有效的二阶依赖.

当前实现对此做了 fallback:

- 如果二阶信息缺失, 则退化为一阶分数 `max(0, -g_e)`.

这样可以保证打分流程的鲁棒性, 不会因为局部 Hessian 缺失而中断整次打分.

## 8. 方法总结

这一实现可以概括为:

1. 通过在 expert 输出上引入缩放变量 `\alpha`, 将 expert 重要性估计转化为 loss 关于 `\alpha` 的局部导数分析.
2. 用 `-g_e + \frac{1}{2} h_e` 近似刻画 expert 的局部二阶贡献.
3. 为降低显存占用, 不再一次性求完整 Hessian 对角, 而是采用 chunked batched autograd 分块提取对角项.
4. 在保持打分定义不变的前提下, 提高了大模型场景下的可运行性.

## 9. 适合论文中的表述重点

如果你要在论文中描述这部分, 建议强调以下几点:

1. 我们使用基于 expert output scaling variable 的二阶局部敏感度估计.
2. 为避免完整 batched Hessian 对角计算带来的显存瓶颈, 我们采用 chunked diagonal Hessian extraction.
3. 该方法只改变二阶项的计算方式, 不改变最终打分公式, 因而在数学目标不变的前提下降低了资源需求.
