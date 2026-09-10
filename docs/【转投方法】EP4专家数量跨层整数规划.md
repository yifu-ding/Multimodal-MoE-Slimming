#【转投方法】EP4专家数量跨层整数规划

## 1. 目标与基本结论

设模型包含 $L$ 个 MoE 层（当前模型约 60--64 层），每层有 $E=128$ 个 expert。将剪枝后的中间宽度量化为

$$
\mathcal M=\{768,512,384,256,0\},
$$

并部署到 4 张 GPU。目标是同时满足：

1. 每层 expert 总数为 128；
2. 所有 MoE 层的总中间宽度满足全局剪枝预算；
3. 每层内部严格“一张 GPU 一个宽度档”，但跨层轮换宽度档，使每张物理 GPU 的累计权重负载在 tolerance 内；
4. 宽度为 0 的 expert 不存放权重，也不进入 dispatch 和 GEMM。

这是一个小规模整数规划问题。由于当前 fused MoE kernel 要求**同一层、同一张 GPU 上的 local experts 具有相同的中间宽度**，所以每层必须把 768/512/384/256 四个非零档分别交给 4 张 GPU，不能在单卡上混放多个宽度后仍使用同一次 fused MoE 调用。

## 2. 显存负载的计算方式

对中间宽度为 $m_e$ 的 gated FFN，gate/up/down 三个 Linear 共有约

$$
P_e=3d_{\mathrm{model}}m_e
$$

个参数。因此，在 $d_{\mathrm{model}}$ 和 dtype 相同时，expert 权重显存与 $m_e$ **线性**相关，不是与 $m_e^2$ 相关。可将 $m_e$ 直接用作静态权重负载的无量纲代理。

若某个 batch 中有 $T_e$ 个 token 路由到 expert $e$，则其中间激活及部分 kernel workspace 可近似写为

$$
A_e\approx C_{\mathrm{act}}m_eT_e.
$$

这只是动态负载代理；真实峰值还受 padding、token capacity、kernel 调度和 all-to-all buffer 影响，所以最终需要用 backend 实测校正。

## 3. 第一阶段：全局 sensitivity-aware 宽度分配

### 3.1 剪枝预算约束

令 $y_{\ell,e,k}\in\{0,1\}$ 表示第 $\ell$ 层 expert $e$ 是否使用档位 $m_k$，并令

$$
n_{\ell,k}=\sum_{e=1}^{128}y_{\ell,e,k}.
$$

每个 expert 只选择一个档位：

$$
\sum_k y_{\ell,e,k}=1,\qquad \forall \ell,e.
$$

完整模型的 expert 总宽度为

$$
B_0=L\times128\times768.
$$

对目标剪枝率 $r$，全局期望保留预算为

$$
B^*=(1-r)B_0.
$$

因为所有非零宽度都是 128 的倍数，定义整数负载单位

$$
u_m=\frac{m}{128}\in\{6,4,3,2,0\}.
$$

于是全局预算约束可写为

$$
\sum_{\ell=1}^{L}\sum_{e=1}^{128}\sum_k u_ky_{\ell,e,k}=U,
\qquad
U=\operatorname{round}\!\left(\frac{B^*}{128}\right).
$$

为了保证每层都能构造四个 fused group，默认要求

$$
n_{\ell,k}\ge1,\qquad
\forall \ell,\ m_k\in\{768,512,384,256\}.
$$

宽度为 0 的档位不要求出现；如果实验必须包含完整 expert removal，再增加全局约束 $\sum_\ell n_{\ell,0}\ge1$，或者设置每层最小删除数。

### 3.2 30% 全局预算的离散化

用 $u=m/128$ 作为单位时，完整模型预算为 $6\times128\times L=768L$。30% 剪枝的目标为

$$
U^*=0.7\times768L.
$$

当 $L=60$ 时，$U^*=32256$，可以精确实现 30%；当 $L=64$ 时，$U^*=34406.4$，取最近整数 $U=34406$，实际剪枝率为

$$
r_{\mathrm{actual}}
=1-\frac{34406}{64\times768}
=30.0008\%.
$$

因此默认应在所有层上一次性离散化全局预算，而不是让每一层分别把 537.6 舍入为 538；后者会积累系统性取整误差。

### 3.3 为什么档位数量不唯一

仅给定 expert 总数和全局剪枝预算时，$n_{\ell,k}$ 有大量可行解。档位数量不应由手工比例决定，而应由 sensitivity 目标函数产生：

$$
\min_y
\sum_{\ell=1}^{L}\sum_{e=1}^{128}\sum_k
c_{\ell,e,k}y_{\ell,e,k}.
$$

这里 $c_{\ell,e,k}$ 是把 expert $(\ell,e)$ 剪到宽度 $m_k$ 的代理损失。求解后，$n_{\ell,k}=\sum_e y_{\ell,e,k}$ 自然给出每一层各档 expert 数量。该目标只声称在给定 sensitivity/proxy cost 下最优，不声称下游任务真实精度全局最优。

### 3.4 如何从 sensitivity 构造 $c_{\ell,e,k}$

最推荐的做法是为每个 expert 直接构造五档代理损失表。先根据已有 channel importance 为 expert $(\ell,e)$ 构造 768/512/384/256/0 五个嵌套 channel mask，再在同一个 calibration set 上计算

$$
c_{\ell,e,k}
=\mathbb E_x\left[
\left\|f_{\ell,e}(x)-f_{\ell,e}^{(m_k)}(x)\right\|_2^2
\right],
$$

也可以使用二阶 Taylor/Hessian 预测损失或被删除 channel importance 之和。这里不运行下游任务精度，只计算结构化剪枝的代理损失。应令 $c_{\ell,e,768}=0$，并对有噪声的估计做单调化：

$$
c_{\ell,e,768}\le c_{\ell,e,512}
\le c_{\ell,e,384}\le c_{\ell,e,256}\le c_{\ell,e,0}.
$$

当前另有独立的 layer sensitivity 系数，记为 $a_\ell>0$。令 $v_{\ell,e}$ 是层内 expert sensitivity 原始分数，先在每层内部变换为非负值并归一化到均值 1：

$$
\widehat s_{\ell,e}
=\frac{v_{\ell,e}+\epsilon}
{\frac1{128}\sum_{j=1}^{128}(v_{\ell,j}+\epsilon)},
\qquad
\frac1{128}\sum_e\widehat s_{\ell,e}=1.
$$

如果原始分数可能为负，应先做平移、softplus 或其他保持排序的正值变换。随后组合层级和 expert 级 sensitivity：

$$
s_{\ell,e}=a_\ell\widehat s_{\ell,e}.
$$

这样 $a_\ell$ 决定第 $\ell$ 层整体对剪枝的敏感程度和获得全局预算的倾向，$\widehat s_{\ell,e}$ 只描述该层内部 expert 的相对顺序与差异。基于标量组合分数时，仍需显式假设“损失如何随剪枝比例增长”。一个简单默认值是

$$
c_{\ell,e,k}
=s_{\ell,e}
\left(\frac{768-m_k}{768}\right)^p,
\qquad p=2.
$$

$p=1$ 容易产生偏极端的 768/0 分配；$p>1$ 表示同一 expert 被继续大幅剪枝时边际损失增加，因此会更多使用中间档。建议对 $p\in\{1,2,3\}$ 做小规模 ablation。只有标量 $s_{\ell,e}$ 时，不可能在没有额外假设的情况下唯一恢复五档损失曲线。

如果使用五档 proxy cost $\widehat c_{\ell,e,k}$，且该 cost 只在层内可比，则同样先做层内尺度归一化，再令 $c_{\ell,e,k}=a_\ell\widehat c_{\ell,e,k}$。如果 Hessian/Taylor cost 本身已经在统一 loss 和采样口径下跨层可比，则不应再次乘 $a_\ell$，避免重复计入层敏感度。

### 3.5 全局 0-1 整数规划

将每层 expert 按 sensitivity 排序，记排序后的索引为 $\pi_{\ell,1},\ldots,\pi_{\ell,128}$，其中

$$
s_{\ell,\pi_{\ell,1}}\ge\cdots\ge
s_{\ell,\pi_{\ell,128}}.
$$

完整优化问题为

$$
\min_y
\sum_{\ell=1}^{L}\sum_{e=1}^{128}\sum_k
c_{\ell,e,k}y_{\ell,e,k},
$$

subject to

$$
\sum_k y_{\ell,e,k}=1,\qquad\forall\ell,e,
$$

$$
\sum_{\ell,e,k}u_ky_{\ell,e,k}=U,
$$

$$
\sum_k m_ky_{\ell,\pi_{\ell,i},k}
\ge
\sum_k m_ky_{\ell,\pi_{\ell,i+1},k},
\qquad\forall\ell,\ i=1,\ldots,127,
$$

$$
n_{\ell,k}=\sum_e y_{\ell,e,k}\ge1,
\qquad\forall\ell,\ m_k>0.
$$

这些约束分别保证唯一档位、全局预算守恒、敏感度顺序守恒，以及每层四个非零 fused group 都存在。$n_{\ell,0}$ 可以为 0；若必须展示完整 expert removal，可增加 $\sum_\ell n_{\ell,0}\ge1$。

由于 $c_{\ell,e,k}$ 都在求解前计算完成，目标函数和约束对二进制变量 $y$ 都是线性的。以 $L=64$ 为例，只有 $64\times128\times5=40960$ 个 $y$ 变量，使用 Gurobi、SCIP 或 OR-Tools CP-SAT 都是合理规模。CP-SAT 需要把浮点 $c$ 乘固定比例后取整。

单调约束意味着每层排序后的 expert 最终形成 768、512、384、256、0 五个连续分段。也可以据此改写为“每层选择四个切分点”，再用跨层 knapsack/DP 分配总预算；但第一版直接实现上述 0-1 模型更清楚，也更容易加入其他约束。

### 3.6 可选的连续目标宽度匹配

如果更希望直接匹配 sensitivity 的相对大小，可将全局归一化分数记为 $q_{\ell,e}$，通过 water-filling 得到

$$
\widetilde m_{\ell,e}=\min(768,\lambda q_{\ell,e}),
\qquad
\sum_{\ell,e}\widetilde m_{\ell,e}=128U,
$$

再定义

$$
c_{\ell,e,k}
=\left(\frac{m_k-\widetilde m_{\ell,e}}{768}\right)^2.
$$

这是另一种 $c_{\ell,e,k}$ 构造方式，不应与前面的剪枝损失 cost 重复叠加。

### 3.7 第一阶段求解伪代码

```text
input:
    layer_sensitivity a[L]
    expert_sensitivity v[L, 128]
    optional proxy_cost c[L, 128, 5]
    target pruning ratio r
    widths = [768, 512, 384, 256, 0]
    units = [6, 4, 3, 2, 0]

U = round((1 - r) * L * 128 * 6)

expert_relative[l] = positive_transform(v[l])
expert_relative[l] /= mean(expert_relative[l])
s[l,e] = a[l] * expert_relative[l,e]

if proxy_cost is unavailable:
    c[l, e, k] = s[l, e] * ((768 - widths[k]) / 768) ** 2

create binary y[l, e, k]
for each (l, e):
    add sum_k y[l, e, k] == 1
add sum_{l,e,k} units[k] * y[l,e,k] == U

for each layer l and each nonzero tier k:
    add sum_e y[l, e, k] >= 1

for each layer l:
    order = argsort(s[l], descending=True)
    for adjacent experts (i, i+1) in order:
        add assigned_width(l, i) >= assigned_width(l, i+1)

minimize sum_{l,e,k} c[l,e,k] * y[l,e,k]
solve width-allocation model

expert_width[l,e] = sum_k widths[k] * y[l,e,k]
count[l,k] = sum_e y[l,e,k]

# Stage 2: widths are fixed. Assign each layer's four groups to four GPUs.
create binary placement p[l,k,g] for nonzero tiers
for each (l,k): add sum_g p[l,k,g] == 1
for each (l,g): add sum_k p[l,k,g] == 1

gpu_weight[g] = sum_{l,k} p[l,k,g] * widths[k] * count[l,k]
mean_weight = sum_g gpu_weight[g] / 4
add abs(gpu_weight[g] - mean_weight) <= tolerance * mean_weight
minimize max_g abs(gpu_weight[g] - mean_weight)
solve cross-layer placement model

return expert_width, count, placement, actual_pruning_ratio, gpu_weight
```

第一阶段只优化 sensitivity 代理并满足全局预算；第二阶段不改变任何 expert 宽度，只在每层的 24 种合法宽度组排列中选择一种，因此不会损害第一阶段的最优目标值。

## 4. 第二阶段：跨层放置宽度组（默认 setting）

### 4.1 fused MoE 对 placement 的硬约束

对每个 MoE 层 $\ell$，四个非零宽度档各形成一个 fused expert group。令 $p_{\ell,k,g}\in\{0,1\}$ 表示层 $\ell$ 的宽度档 $k$ 是否放到 GPU $g$，则

$$
\sum_g p_{\ell,k,g}=1,\qquad
\sum_{k:m_k>0}p_{\ell,k,g}=1.
$$

即每个非零宽度组只放到一张卡，每张卡在该层只接收一个宽度组。宽度为 0 的 expert 不需要放置。在第一阶段已确定 $n_{\ell,k}$ 后，层内静态负载为

$$
W_{\ell,g}=\sum_{k:m_k>0}p_{\ell,k,g}m_kn_{\ell,k}.
$$

### 4.2 全局累计权重与 tolerance

fused MoE 只要求“同一层、同一张卡”内的 expert 宽度一致，不要求某张物理 GPU 在所有层都负责同一个宽度档。因此可以在不同层轮换四个宽度组到物理 GPU 的映射。整个模型的静态负载为

$$
W_g^{\mathrm{model}}=\sum_\ell W_{\ell,g},
\qquad
\bar W^{\mathrm{model}}=\frac14\sum_g W_g^{\mathrm{model}}.
$$

对跨层静态 tolerance $\tau_w$，默认要求

$$
\left|W_g^{\mathrm{model}}-\bar W^{\mathrm{model}}\right|
\le\tau_w\bar W^{\mathrm{model}}.
$$

具体求解时引入非负变量 $d$：

$$
-d\le W_g^{\mathrm{model}}-\bar W^{\mathrm{model}}\le d,
\qquad\forall g,
$$

$$
d\le\tau_w\bar W^{\mathrm{model}},
\qquad
\min_p d.
$$

由于第一阶段已经固定 $n_{\ell,k}$，$m_kn_{\ell,k}$ 在该模型中是常数，所以 $W_g^{\mathrm{model}}$ 对二进制 placement 变量 $p_{\ell,k,g}$ 是线性的。该模型只有约 $L\times4\times4$ 个二进制变量；$L=64$ 时约为 1024 个。

若每层的四个档位负载相同，每 4 层做一次循环轮换，则每张 GPU 恰好各承担一次四种档位，跨层权重可完全均衡。若各层 $n_{\ell,k}$ 不同，则在第一阶段结果固定后，对每层 24 种档位到 GPU 的排列做一个小型 assignment/DP 即可最小化跨层最大累计负载。

默认求解顺序是：先在全局预算下得到 sensitivity 最优的 $y_{\ell,e,k}$，再固定 $n_{\ell,k}$ 求解 $p_{\ell,k,g}$。如果第二阶段无法满足给定 $\tau_w$，再采用 lexicographic 方式回到第一阶段：首先限制 sensitivity objective 不超过最优值的 $(1+\delta)$，然后在该近优解集合内联合调整各层档位数量和跨层 placement。

跨层轮换能平衡模型常驻权重显存和长时间累计 GPU 工作量。它不保证单层四组的计算时间相等，但默认 setting 不再对单层施加权重均衡约束；运行时 straggler 留到第 5 节通过实测评估。

## 5. 加入路由 token 后的动态负载均衡（后续阶段）

### 5.1 需要收集的统计量

在代表性校准集上，对每层、每个 expert 和每个 micro-batch 记录：

- 路由 token 数 $T_{\ell,e,b}$；
- 路由概率的 mean、P95 和 max；
- 每层 all-to-all 的发送/接收 token 数；
- 每张 GPU 的 allocated/reserved peak memory；
- expert GEMM、dispatch、all-to-all 和整层延迟。

不建议只使用全校准集的 token 总数，因为总数会掩盖 batch 级峰值。放置时至少应使用 P95，并用 max 做压力测试。

### 5.2 动态约束

不应先对每个 expert 单独取 P95 再求和，因为这会忽略 expert 热度之间的相关性。令 $\mathcal E_{\ell,g}$ 表示根据宽度组映射 $p_{\ell,k,g}$ 放在 GPU $g$ 的整组 expert。对层 $\ell$、GPU $g$ 和校准 micro-batch $b$，先定义 batch 级负载

$$
A_{\ell,g,b}=\sum_{e\in\mathcal E_{\ell,g}}m_{\ell,e}T_{\ell,e,b},
$$

以及通信 token 负载

$$
R_{\ell,g,b}=\sum_{e\in\mathcal E_{\ell,g}}T_{\ell,e,b}.
$$

再在放置完成后沿 $b$ 统计 batch 级负载的 P95：

$$
A_{\ell,g}^{95}=\operatorname{P95}_b(A_{\ell,g,b}),\qquad
R_{\ell,g}^{95}=\operatorname{P95}_b(R_{\ell,g,b}),
$$

$$
\bar A_\ell^{95}=\frac14\sum_g A_{\ell,g}^{95},\qquad
\bar R_\ell^{95}=\frac14\sum_g R_{\ell,g}^{95}.
$$

求解时可直接对一组代表性 batch scenario 最小化最坏负载，然后在独立留出 batch 上计算 P95，避免对校准负载过拟合。

分别设定动态 tolerance $\tau_a$ 和通信 tolerance $\tau_r$：

$$
|A_{\ell,g}^{95}-\bar A_{\ell}^{95}|\le\tau_a\bar A_{\ell}^{95},
$$

$$
|R_{\ell,g}^{95}-\bar R_{\ell}^{95}|\le\tau_r\bar R_{\ell}^{95}.
$$

如果要使用单一目标，可对三类负载先做归一化，再最小化最坏 GPU 的综合负载：

$$
\min\max_g\left[
\lambda_w\frac{W_g^{\mathrm{model}}}{\bar W^{\mathrm{model}}}+
\lambda_a\max_\ell\frac{A_{\ell,g}^{95}}{\bar A_\ell^{95}}+
\lambda_r\max_\ell\frac{R_{\ell,g}^{95}}{\bar R_\ell^{95}}
\right].
$$

这部分暂不加入第一版的 sensitivity-aware 宽度分配。待 vLLM 非均匀 EP 跑通后，再根据实测设定动态负载 tolerance $\tau_a$ 和通信 tolerance $\tau_r$；不应在没有实测数据时预先宣称某个固定数值。

## 6. 完整规划与校验流程

1. **确定预算：** 输入目标剪枝率 $r$，将连续预算舍入到最近的 128 倍数，记录实际剪枝率。
2. **读入 sensitivity：** 对每层 128 个 expert 取得 $s_e$，做稳健归一化并按降序排列。
3. **生成连续目标：** 通过 water-filling 计算 $\widetilde m_e$，使连续宽度之和等于保留预算。
4. **离散档位规划：** 在精确预算、排序单调和四个非零档都存在的约束下，最小化 sensitivity 匹配代价，输出每个 expert 的宽度及 $n_{\ell,k}$。
5. **构建 fused groups：** 每层按 768/512/384/256 将非零 expert 建成四个同宽度 group，删除宽度为 0 的 expert 权重。
6. **跨层静态放置：** 默认求解各层四个 group 到物理 GPU 的轮换映射，最小化整个模型的最大累计权重负载并满足 $\tau_w$。
7. **后续校准：** vLLM 实现跑通后，再收集 batch 级 token、peak memory、GEMM 和 all-to-all 指标，必要时把动态代价加回宽度规划。
8. **留出验收：** 报告精度、throughput、TTFT、TPOT、P50/P95 latency、逐卡 peak memory 和 OOM 情况。

## 7. 实现建议

- **第一版：** 实现全局 sensitivity-aware 档位规划和跨层静态 placement，不引入运行时 token 负载。
- **求解方式：** 优先使用排序后的连续分段枚举/DP，并用 OR-Tools CP-SAT、SCIP 或 Gurobi 的解作为正确性对照。
- **fused group：** 每层每张卡只保留一个中间宽度，不再使用“单卡混放多宽度”假设。
- **优先保持宽度：** 后续校准先优化跨层宽度组到物理 GPU 的映射，不因短期 token 波动反复改变 expert 宽度。
- **结果可复现：** 保存 sensitivity 归一化参数、$\gamma$、档位代价矩阵、每层的 `global_expert_id -> (width, gpu_id)` 映射、求解器 seed 和剪枝预算。

## 8. Codex 补充：对算法语义的理解

> [!NOTE]
> 本节及后续第 9--15 节为 Codex 于 2026-09-10 在阅读当前仓库源码后追加的理解、代码审计和实施计划，不是原设计文档的一部分。

这个方法的核心不是“给 4 张卡各固定一个宽度”，而是以层为单位选择一个档位到 EP rank 的置换（permutation）：

```text
layer 0: rank 0/1/2/3 <- 768/512/384/256
layer 1: rank 0/1/2/3 <- 256/384/512/768
layer 2: rank 0/1/2/3 <- 384/768/256/512
...
```

因此，需要同时保持下列三个不变量：

1. 对任意层，`768/512/384/256` 四个非零档位与 4 个 EP rank 一一对应；
2. 对任意层的任意 rank，本地 expert 只有一种宽度，但 expert 数量可以与其他 rank 不同；
3. 对不同层，同一物理 rank 接收哪个档位没有顺序约束，只优化全模型跨层累计负载。

第一阶段回答“每个 expert 保留多宽”，第二阶段回答“每层的四个宽度组各放在哪个 rank”。第二阶段不得改变第一阶段的 expert 宽度，因而不会改变 sensitivity objective 或剪枝预算。

宽度 0 是删除状态，不是第五个 fused group。它的全局 expert ID 仍需在 router 语义中保留，但对应的 `rank/local_id` 都是 `-1`，不加载权重，不参与 dispatch 和 GEMM。

## 9. Codex 补充：现有源码审计

### 9.1 已经可以直接复用的部分

1. [`src/calibration/score_accumulator.py`](../src/calibration/score_accumulator.py) 已保存规划需要的基础信号：
   - `channel_scores[metric][layer][expert]`，可组成 `[L,E,I]`；
   - `expert_scores[metric][layer][expert]`，可组成 `[L,E]`；
   - `layerwise_loss` 和 `layerwise_second_order_sum`；
   - `metadata.layers`、`layer_to_num_experts` 和 `layer_to_num_channels`。
2. [`src/generate_mask/stages/prepare_scores.py`](../src/generate_mask/stages/prepare_scores.py) 已能把嵌套的 score 转换为密集 tensor，并正确处理 `layers` 在 top-level 或 `metadata` 中的新旧格式。
3. 现有 channel score 已包含 `3proj_second_order`、`3proj_act` 等指标。对一个 expert 将 channel score 降序排序后，保留 top-$m$ 即可为每个候选档位生成嵌套 mask。
4. [`src/generate_mask/planners/inter_layer/algo/loss_based.py`](../src/generate_mask/planners/inter_layer/algo/loss_based.py) 的 layer loss 非负化、平滑和均值归一化逻辑可以抽取成新 cost builder 的可选 layer scaling policy。
5. 当前 `maes` 环境已有 `scipy==1.17.1` 和 `scipy.optimize.milp`；没有 OR-Tools。第一版可以直接用 SciPy/HiGHS 实现 MILP，不必新增重依赖。

### 9.2 不能直接复用的部分

1. [`src/generate_mask/pipeline.py`](../src/generate_mask/pipeline.py) 当前的契约是输出任意 boolean channel mask，不保证每个 expert 宽度属于 `{768,512,384,256,0}`。
2. 现有 inter-layer planner 先生成浮点层保留率，超过 1 时直接 clamp，只警告均值偏差；它不能保证全局离散预算精确相等。
3. `build_masks_expertwise()` 使用 `max(1, ceil(...))`，因此不能产生宽度 0；后续 `adjust_masks()` 还可能把已量化的宽度再次改成非法档位。
4. `build_masks_globally()` 对按权重分配的 $K_E$ 直接取 floor 且没有回填余数，无法守住精确预算；当前实现还把 layer 循环写成了 `for lid in len(scores)`，因此不能作为新 Stage 1 的基础。
5. [`src/prune.py`](../src/prune.py) 适合单机 PyTorch 精度验证，不是 EP4 runtime：
   - Qwen3 fused experts 被转换为逐 expert `ModuleList`，无法直接满足 rank-local FusedMoE 的统一 shape 契约；
   - expert 宽度为 0 时会裁剪 gate 并重编 expert ID，这与 EP4 所需的“router 仍为 128 维，删除 expert 用 `-1` 失效”语义直接冲突。
6. 当前仓库只有 vLLM baseline 入口和设计文档，尚无 pruning plan compiler、per-layer placement loader 或非均匀 EP runtime 实现。

### 9.3 当前数据和模型假设中需要先解决的问题

1. 目标 `Qwen/Qwen3-VL-30B-A3B-Instruct` 在仓库已核对的结构是 48 个 MoE 层、128 个 expert、expert width 768，不是文档开头估计的 60--64 层。新代码必须从 score metadata 和 model config 读取并交叉校验 `L/E/I`，不允许写死。
2. 当前 `storage` 是一个目标已不存在的符号链接，仓库中脚本所记录的 Qwen3 `scores.pt` 无法读取。因此本次只能核对 score schema 和产生逻辑，尚未对真实 score 的 key、shape、NaN/Inf、零值率和数值尺度做 data audit。
3. [`docs/vllm_qwen3_vl_ep_pruning_plan.md`](vllm_qwen3_vl_ep_pruning_plan.md) 第 8 节曾计划在宽度分配时同时优化路由计算负载；本文的默认算法则明确把路由动态负载放到后续阶段。实现 v1 时应以本文为准：宽度只由 sensitivity/proxy cost 决定，placement 只优化跨层静态权重负载。
4. 本文第 3.4 节的 proxy cost 与第 3.6 节的 water-filling 是两种备选 cost 定义，不能先后叠加。第 6 节第 3 步应视所选 cost mode 而定，不是所有运行都必须 water-filling。

## 10. Codex 补充：建议的模块边界

不建议把整数规划作为又一个 `intra_layer_method` 塞入现有 `generate_masks()`。这个方法同时产生 width allocation、placement、global/local expert mapping 和可复现性元数据，其输出契约明显大于一个 boolean mask。

建议拆成与 GPU/vLLM 无关的 CPU planner 和只消费计划产物的 runtime adapter：

```text
scores.pt + model config
        |
        v
score adapter -> tier cost + nested candidate masks
        |
        v
width MILP -> expert_width + per-layer tier counts
        |
        v
placement MILP -> tier_to_ep_rank
        |
        v
validated EP4 plan artifact
        |
        +--> offline report / PyTorch reference
        |
        `--> vLLM plan loader / weight loader / dispatcher
```

建议的文件布局：

```text
src/calibration/score_io.py             # 新旧 scores.pt 的公共读取与 shape 校验
src/ep4_planner/schema.py               # 配置、中间结果、最终 plan 的 typed schema
src/ep4_planner/costs.py                # 五档 cost、嵌套 channel mask、layer scaling
src/ep4_planner/width_solver.py         # Stage 1 MILP
src/ep4_planner/placement_solver.py     # Stage 2 MILP
src/ep4_planner/mapping.py              # global/rank/local ID 的双射映射
src/ep4_planner/artifact.py             # save/load/hash/schema migration
src/ep4_planner/validate.py             # 预算、档位、映射、负载统一校验
scripts/compile_ep4_plan.py             # 唯一 CLI 入口
tests/ep4_planner/                      # 不需 GPU 的规划器测试
src/vllm_integration/                   # 后续 runtime，不反向依赖 planner 内部实现
```

`prepare_scores.py` 应改为使用新的公共 `score_io` helper，保持它的现有 API 和返回值不变。这样 EP4 planner 不依赖旧 mask pipeline 的 method dispatch，也不会复制一套不兼容的 score loader。

## 11. Codex 补充：规划器的具体实现契约

### 11.1 输入校验与预算

planner 首先从 score metadata 获得真实 model layer ID 列表，再与 model config 交叉校验：

- 所有规划层的 expert 数相同，当前目标是 128；
- 所有 expert 的原始宽度为 768；
- `tiers=[768,512,384,256,0]` 严格递减、不重复，非零档位数等于 `ep_size=4`；
- score 必须 finite，shape 必须与 metadata 完全一致；
- positional layer index 只用于 tensor 运算，artifact 中始终保存原始 model layer ID。

全局预算使用 `round((1-r)*L*E*I/128)` 个整数单位，并在求解前检查可行区间。由于要求每层四个非零档位都至少有一个 expert，一层的最小单位数是 `6+4+3+2=15`，最大单位数是 `6*(E-3)+4+3+2`。超出区间或离散预算不可达时必须 fail fast，不能静默改剪枝率。

### 11.2 五档 cost 和 channel mask

v1 建议默认使用已有的 `3proj_second_order` 或用户显式指定的 channel metric：

1. 对每个 expert 的 768 个 channel 按 score 降序稳定排序，分数相同时按 channel ID 打破平局；
2. 宽度 $m$ 的候选 mask 保留前 $m$ 个 channel，因此五档 mask 天然嵌套；
3. 定义 `cost[l,e,m]` 为被删除 channel 的非负 score 之和，并用 cumulative sum 一次生成全部档位；
4. 如果要使用 `layerwise_second_order_sum` 调制跨层尺度，必须通过显式 `layer_scale_mode` 开启，保存归一化后的 $a_l$；不能在 channel cost 已可跨层比较时再隐式乘一次。

water-filling 作为单独的 `cost_mode=target_width` 保留，与 `cost_mode=removed_channel_score` 互斥。所选 mode、score key、非负变换、layer scaling 和完整 cost tensor 的 hash 都要记录到 artifact。

用于宽度单调约束的 expert sensitivity 也必须显式指定 `expert_order_source`。建议 v1 使用同一 cost table 的完全删除代价 `cost[l,e,0]` 排序，使目标和顺序约束来自同一信号。如果实验要改用 `expert_scores.second_attr`，应作为明确的 ablation 配置并记录到 artifact，不能因 key 缺失而静默 fallback。

### 11.3 Stage 1：width MILP

按本文第 3.5 节直接建立 binary MILP。对 Qwen3 的实际 48 层，变量数是 `48*128*5=30720`，而不是 64 层示例的 40960。

约束必须包含：

- 每个 expert 恰好选一档；
- 全局整数预算精确相等；
- 每层四个非零档位各至少一个 expert；
- 按专家 sensitivity 降序排列后的宽度单调不增；
- 相同 sensitivity 用 global expert ID 稳定打破平局，确保重复运行一致。

用 SciPy/HiGHS 求解后不直接信任浮点结果，而是先把 `y` 解析成整数 width，再用独立 validator 重新计算每条约束和 actual pruning ratio。如果求解器非 optimal，或任一 expert 没有唯一档位，编译失败。

### 11.4 Stage 2：cross-layer placement MILP

Stage 2 的输入是固定的 `count[l,tier]`。每层只有 24 种合法 permutation，但跨层组合仍由 MILP 统一求解，优化

$$
\min \max_g \left|\sum_l m_{l,g}n_{l,g}-\bar W\right|.
$$

在 v1 四个 rank 的非 expert 基础负载被视为对称时，固定第一层为 identity permutation 可用于消除 GPU label 对称性，不损失最优性。如果后续把已有的 rank-specific 显存或计算基线加入目标，就不能再固定第一层。求解器先最小化实际最大偏差，再由 validator 检查是否小于 `tau_w`；这样 tolerance 不满足时仍能报告“可达的最佳偏差”，而不是只返回 infeasible。

v1 不在这一阶段加入 token frequency、P95 activation 或 all-to-all 负载。如果静态 tolerance 不满足，第一版应停止并输出诊断，不得静默回改 Stage 1 的宽度。近优联合求解作为后续显式 mode 再加入。

### 11.5 映射和 channel 选择

placement 确定后，按 global expert ID 升序为每个 `(layer,rank)` 生成稳定 local ID。必须生成并校验：

```text
expert_width[layer, global_expert_id]
tier_to_ep_rank[layer, tier]
expert_to_rank[layer, global_expert_id]
expert_to_local_id[layer, global_expert_id]
local_to_global[layer, rank, local_expert_id]
channel_mask[layer, global_expert_id, channel_id]
```

channel mask 必须直接从生成 cost 时的同一份稳定排序取 top-width，不能在求解后再调用旧 `trim_masks_to_layer_budget()` 或 `adjust_masks()`，否则 cost 与最终裁剪通道不再一致。

## 12. Codex 补充：plan artifact 建议

建议一次编译生成一个目录，而不是一个无版本的 `masks.pt`：

```text
ep4_plan/
  manifest.json       # schema、provenance、宽度、placement、mapping、统计
  tensors.pt          # bool channel mask 及其他密集 CPU tensor
  report.md           # 便于人工审计的逐层/逐 rank 报表
```

`manifest.json` 至少保存：

- `schema_version`、模型 ID/revision/config hash；
- scores 绝对路径仅用于 provenance，同时必须保存 content SHA256；
- 所有 cost 配置、tier 列表、target/actual pruning ratio；
- 原始 model layer IDs，不只是 `0..L-1` 的紧凑位置；
- `expert_width`、tier counts、tier-to-rank permutation 和 global/local mapping；
- 每层每 rank 的 expert 数和 width load，以及全模型累计负载；
- solver 名称/版本、status、objective、MIP gap、seed 和运行时间；
- `tensors.pt` 的 SHA256。

`tensors.pt` 只保存 tensor 和基本容器，loader 使用 `weights_only=True`。加载时重做全部 invariant 校验，不把“成功反序列化”当作计划有效的证明。

## 13. Codex 补充：按阶段修改代码的计划

### Phase 0：恢复并审计真实 score

- 修复 `storage` 的数据路径，或通过 CLI 显式传入真实 `scores.pt`；
- 输出 score keys、`L/E/I`、每个 metric 的 min/max/mean/零值率、NaN/Inf 和 layer scale；
- 确认 Qwen3 的完整 48 层都存在，部分层 calibration 的产物不能误当成全模型计划输入。

验收：一个只读 audit 命令能对所选 scores 给出完整可用性结论。

### Phase 1：score adapter、schema 和 cost table

- 抽取公共 `score_io`，保持现有 `generate_masks()` 行为不变；
- 实现 typed planner config/result；
- 实现嵌套候选 mask 和五档 cost；
- 实现预算可行性和 score/model shape 校验。

验收：对任意 expert，`cost(768)=0`、cost 随宽度减小而非降，mask 严格嵌套且 mask sum 等于档位。

### Phase 2：Stage 1 width solver

- 用 `scipy.optimize.milp` 实现本文完整约束；
- 实现 deterministic tie-breaking 和求解后独立校验；
- 输出 `expert_width`、tier counts、objective 和 actual pruning ratio。

验收：预算整数单位完全相等，每层四档均非空，宽度排序约束完全满足。

### Phase 3：Stage 2 placement、mapping 和 artifact

- 实现跨层静态负载 MILP；
- 生成每层 tier permutation 和 global/local mapping；
- 实现 versioned artifact、SHA256、validator 和 `report.md`；
- 提供 `scripts/compile_ep4_plan.py`。

验收：每层 tier/rank 均为双射，每 rank 在同一层只有一种宽度，删除 expert 的 mapping 均为 `-1`，全局负载偏差报告与重算一致。

### Phase 4：与旧 PyTorch 路径的离线数值验证

- 只把最终 `channel_mask` 适配为旧 `apply_structural_pruning()` 可读的 layer mask，用于单机精度和权重切片校验；
- 明确标记该路径不验证 EP placement，也不作为 width-0 router 语义的 reference；
- 对首层、中间层和末层的各档 expert 核对 gate/up/down 的 channel slicing。

验收：保留权重与原 checkpoint 按 channel index 取值逐元素一致。

### Phase 5：vLLM EP4 runtime

- 在 `src/vllm_integration/` 加载已经验证的 plan，不在 runtime 重新求解；
- 保留 replicated 128-way router，top-k 后将删除 expert 的 ID/weight 改为 `-1/0`；
- 按每层 mapping 进行 dispatch，每 rank 只加载本地 expert 的裁剪权重；
- 将该层本地同宽 expert pack 成 vLLM FusedMoE 所需 shape；
- 与同一 vLLM 版本下的 baseline 做 BF16 reference、四卡 dispatch 和端到端评测。

验收：本文第 1 节和 `vllm_qwen3_vl_ep_pruning_plan.md` 第 17 节的完成标准全部通过。

## 14. Codex 补充：最小必要测试集

1. **Cost 测试：** 人工构造 channel score，验证五档 cost、嵌套 mask、tie-breaking 和非负变换。
2. **小规模最优性测试：** 对小 `L/E/tier` 例子枚举全部解，与 width MILP objective 逐项对比。
3. **预算测试：** 精确可达、舍入后可达、四非零档约束导致不可达三类情况。
4. **Placement 测试：** 构造两层，确认能生成类似 identity/reverse 的不同 permutation；再用 4 层对称例子验证完全均衡。
5. **Mapping 测试：** active expert 的 global/local mapping 是双射，width-0 expert 没有 rank/local ID，逐 rank 宽度唯一。
6. **Artifact 测试：** round-trip、schema version、hash mismatch、model revision mismatch 和损坏 tensor 都有明确结果。
7. **确定性测试：** 同一输入连续编译两次，width、placement、mapping 和 artifact hash 一致。
8. **真实 score smoke test：** 数据路径恢复后，用 Qwen3 全 48 层 scores 编译 30% plan，验证 actual ratio、求解时间、内存和逐层报表。

## 15. Codex 补充：建议的 v1 默认选择

为了让第一版的数学目标、代码和实验解释保持一致，建议固定以下默认值：

```text
ep_size = 4
tiers = [768, 512, 384, 256, 0]
require_each_active_tier_per_layer = true
cost_mode = removed_channel_score
channel_metric = 3proj_second_order
layer_scale_key = layerwise_second_order_sum
layer_scale_mode = explicit_normalize_then_multiply
expert_order_source = full_removal_cost
enforce_monotonic_width_by_sensitivity = true
placement_objective = cross_layer_static_weight_minimax
dynamic_routing_load = disabled
solver = scipy.optimize.milp (HiGHS)
```

其中 `layer_scale_mode` 在真实 score data audit 后必须再确认。如果 `3proj_second_order` 在相同 loss、样本和累加口径下已经可以跨层直接比较，则应设为 `none`，避免重复计入 layer sensitivity。

## 16. Codex 补充：2026-09-10 最新实现决策

> [!IMPORTANT]
> 本节记录最新讨论结果，并覆盖本文前面与之冲突的旧方案。v1 不再对所有 layer/expert/tier 建立全局宽度 MILP；默认活动档位改为 `768/640/512/384`，宽度低于 384 时直接量化为 0（删除 expert）。原 `768/512/384/256` 方案保留为可配置 ablation。

最新流程为：

1. 输入 `layer_sensitivity[L]`、`expert_sensitivity[L,E]`、`scores[L,E,I]` 和 `prune_ratio`，其中 `keep_ratio=1-prune_ratio`；
2. 调用现有 layer score-coverage binary search，根据逐层 sensitivity 分配每层原始保留通道数；
3. 每层调用现有 expert coverage binary search。其搜索变量 alpha 控制每个 expert 需要覆盖的累计 channel score 比例，而不是直接按通道数量比例分配；
4. 将原始 `[L,E]` 通道数圆整到 `{768,640,512,384,0}`，并以 128 为单位修复全局离散预算；
5. 每层四个活动档位必须各出现至少一次，0 档不参与 placement；
6. placement 阶段每层枚举 24 种档位到 EP rank 的 permutation；默认先贪心平衡部分累计负载，再用逐层局部搜索降低最终最大偏差。小规模或离线最优性对照可显式选择 MILP；
7. 若 placement 的相对偏差不超过 `tolerance`，则接受；否则返回当前最佳偏差，并由 strict mode 决定是否报错。

这样 width planning 的主体复杂度是 binary search、排序和贪心档位修复。默认 placement 每层只评估 24 个候选，不会随层数产生组合爆炸。保留的 MILP 对照路径包含约 `L*24+1` 个变量；Qwen3 的 48 层对应约 1153 个变量，但实测仍明显慢于默认方法。

选择 `{768,640,512,384,0}` 的原因是：30% 剪枝时平均目标宽度为 537.6，512 和 640 正好位于目标两侧，量化误差较小；384 作为最低活动宽度比 256 更保守。该选择理论上更有利于保留单个活动 expert 的表达能力，但会增加直接删除弱 expert 的可能性，最终结论仍需与 `{768,512,384,256,0}` 做同预算精度 ablation。
