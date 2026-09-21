 实验 C（服务器上可跑）：greedy vs MILP，填表 6

**你的原问题。** ==这个确实是个问题，但是我们没有办法尝试 M=8，因为现在只有四卡。只有一张八卡的 L20，可能可以尝试一下，在那张八卡 L20上尝试这个。但是我想了一下，这个是不是只需要做 plan 就可以？也就是说，它的 accuracy 和 efficiency 的实测，在端到端模型上的实测其实是不必要的。所以只需要算 M=8时 greedy 和 MILP 的差异结果，也就是它们的 plan 结果有百分之多少的差异就可以，不需要实际去跑端到端模型。如果是这样的话，我们可以马上做这个实验。==

> [!check] 你的判断完全正确——这个实验不需要 GPU
> 放置问题是**纯组合优化**：输入是每层每个 width level 的专家数 n^ℓ_k，输出是一个 rank 分配，目标函数 ΔΦ/Φ̄ 只依赖这些标量。**模型权重、显卡、端到端推理全都不需要。** 而且如原始 review 里提过的，placement 只是排列哪个 rank 放哪组宽度，不改变任何专家宽度和路由，所以 accuracy 恒等——端到端实测确实是多余的。一台笔记本就能跑完 m=4/8/16 的全部对比。


### 实验 C 执行方案（2026-09-18，查过代码后重写）

我把 `/Users/yifuding/LocalFiles/codes/Multimodal-MoE-Slimming` 翻了一遍，**上面"卡点 1 / 卡点 2"全部作废**——代码早就有了，而且比我以为的完整。下面是可以直接照做的方案。

#### 先回答你的四个问题

**1. 已经有 MILP 代码了吗？——有，两个求解器都在。**
文件：`src/generate_mask/ep4_intplan.py`（1511 行）。统一入口是

```python
solve_cross_layer_placement(
    width_counts,            # torch.Tensor [L, m]，每层每个宽度档的专家数
    active_widths,           # 长度 m 的宽度列表，如 (384, 512, 640, 768)
    tolerance=0.01,
    fix_first_layer=True,
    method="greedy",         # 或 "milp"
    max_local_search_passes=100,
)
```
- `method="greedy"` → `_solve_placement_groups_greedy`，就是论文 Algorithm 2（贪心 + coordinate-descent 局部搜索）。
- `method="milp"` → `_solve_placement_groups_milp`，用 **`scipy.optimize.milp`（HiGHS 后端）**，不需要装 pulp。
- 文件头注释已写明设计意图："A greedy pass plus local search balances cumulative rank load; an exact MILP remains available as an offline reference." ——**这个对比实验本来就是预留好的**。

**2. 是不是只用 CPU？——是，完全不碰 GPU。**
依赖只有 numpy / scipy / torch（CPU tensor，代码里显式 `.detach().cpu()`）。输入是一个 `L×m` 整数矩阵，跟模型权重、推理、显存都无关。

> [!warning] 但有一个实际干扰点，必须处理
> HiGHS 和 numpy 默认会**吃满所有 CPU 核**。你的 GPU 实验虽然算力在卡上，但 dataloader / 预处理是 CPU 的，MILP 抢满核会让 GPU 任务掉吞吐。跑之前务必限核：
> ```bash
> export OMP_NUM_THREADS=4
> export MKL_NUM_THREADS=4
> taskset -c 0-3 python scripts/run_placement_ablation.py
> ```
> 限核之后对 GPU 任务基本无感。**但注意**：限核会影响 solving time 的绝对值。两个求解器必须在**同样的限核条件**下测，否则时间对比不公平；论文里也建议注明测量环境（如 "4 CPU threads"）。

**3. 能跑 m=8 或 m=16 吗？——现在的代码不行，而且这个"不行"本身就是结果。**
- **硬性限制**：`_solve_cross_layer_placement_greedy` 里有 `if num_widths <= 0 or num_widths > 7: raise ValueError`，m≥8 直接报错。
- **更本质的原因**：MILP 的建模是"每层枚举全部 m! 个双射，每个双射一个 0-1 变量"（代码第 700 行 `itertools.permutations(range(ep_size))`），二元变量总数是 **L × m!**：

| m | m! | L=48 时的二元变量数 | 可行性 |
|---|---|---|---|
| 4 | 24 | 1,152 | 秒级 |
| 5 | 120 | 5,760 | 很快 |
| 6 | 720 | 34,560 | 分钟级 |
| 7 | 5,040 | 241,920 | 很慢，但能跑 |
| 8 | 40,320 | 1,935,360 | **基本不可解** |
| 16 | 2.09×10¹³ | — | 完全不可能 |

> [!tip] 所以我建议改一下实验设计，这样反而更有说服力
> 不要执着于"m=8 时 greedy 和 MILP 差多少"。**把 m 当横轴，展示 MILP 的崩溃过程**：跑 m ∈ {4, 5, 6, 7}，记录两者的 ΔΦ/Φ̄ 和求解时间。预期结论是"**greedy 的质量一直贴着 MILP，但 MILP 的时间指数爆炸，到 m=7 已经不实用，m=8 直接不可解**"——这恰好**证明了论文为什么必须用 greedy**，比"m=8 时差 0.3%"这种孤立数字强得多。
> 然后 m=8/16 只跑 greedy，报告它的 ΔΦ/Φ̄ 和毫秒级耗时，说明"在 MILP 已不可解的规模上 greedy 仍给出低不平衡度的解"。跑 m≥8 需要先改掉那行 `> 7` 的守卫，并确认 greedy 内部没有别处依赖 m! 枚举（它的局部搜索是 coordinate descent，理论上不受限，但要实测确认）。

> [!check] 你已确认采用这个设计（2026-09-18）
> ==（你的意见）我觉得你可以改成这样的实验设计：限制 CPU 的核数。我们先从4开始，也就是先从 M=4 开始跑，然后逐渐跑 M=5、6、7 这些情况。把 M 当成横轴，我觉得是 OK 的。到 M=8 或者以上的时候，就会发现 MILP 是不是会不可解，或者求解时间会特别长。我觉得这个实验是可以的，我们可以从时间短的开始跑。==
>
> 按你的意见定稿成 **m 递增扫描 + 固定核数**。关键是**从 m=4 开始逐级往上，每级跑完立刻落盘**，这样即使 m=7 卡死，前面几级的数据也已经拿到了，不会白跑。下面的 `run_m_sweep.py` 就是按这个逻辑写的。
>
> **必须设超时。** m=7 的 24 万个二元变量可能跑几十分钟到几小时，m=8 可能永远不返回。所以每次 MILP 调用都要有 wall-clock 上限，超时就记为"未在 T 秒内求解"并继续下一级——**超时本身就是要报告的结果**，不是失败。建议 `MILP_TIMEOUT = 1800`（30 分钟）。
>
> **限核对这个实验是必须的，不只是为了不打扰 GPU 任务。** 求解时间是这张表的核心指标之一，必须在固定且可复现的 CPU 配置下测；如果中途别的任务抢核，时间数据就没有可比性。所以整个扫描**从头到尾用同一组核**跑完，中间不要改 `taskset`。论文里注明"all timings measured on N pinned CPU threads"。

**4. 直接跑这个表格就行吗？——是的，表 6 的六行就是六次调用。**
表 6 的两个自变量是 `p ∈ {30%, 50%}` 和 `L ∈ {8, 24, 48}`，m 固定为 4。L=8/24 就是取 Qwen3-VL-30B 的前 8 / 前 24 层（`width_counts[:L]`），**不需要重新剪枝**。

#### 需要什么数据

只需要一样：**剪枝后的 `width_counts`（`L × m` 整数矩阵）**。由现成脚本产出：

```bash
python scripts/build_ep4_pruning_plan.py \
  --scores <你的校准分数文件> \
  --model "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --prune-ratio 0.3 \
  --output plans/qwen3_p30.json
```
- **模型**：`Qwen/Qwen3-VL-30B-A3B-Instruct`（和表 6 caption 一致），宽度档预设 `(0, 384, 512, 640, 768)`。
- **数据集**：`--scores` 用你主实验同一套（GQA / COCO / M4-Instruct 混合，1024 samples × 2048 tokens）。placement 本身只依赖宽度分布、不依赖数据集，但为了让表 6 与主表自洽，应当用**与主实验完全相同的校准配置**。
- 两个剪枝率各跑一次（`0.3` 和 `0.5`）。

> [!note] 如果服务器上已有跑好的 plan json
> 不用重新生成，直接从 json 读每层每档的专家计数（输出里有 `expert_widths` 字段，按层统计各宽度出现次数就是 `width_counts`），连校准都省了。

#### 建议的驱动脚本

在仓库根目录建 `scripts/run_placement_ablation.py`：

```python
import json, time, torch, numpy as np
from src.generate_mask.ep4_intplan import solve_cross_layer_placement

ACTIVE_WIDTHS = (384, 512, 640, 768)      # Qwen3-VL-30B，去掉 0 档
REPEATS = 5                                # 计时取中位数，避免抖动

def width_counts_from_plan(path, widths):
    """从 plan json 还原 [L, m] 计数矩阵。字段名按实际输出调整。"""
    plan = json.load(open(path))
    per_layer = plan["expert_widths"]      # [L][E] 每个专家的宽度
    counts = torch.zeros(len(per_layer), len(widths), dtype=torch.int64)
    for l, row in enumerate(per_layer):
        for w in row:
            if int(w) == 0:                 # 被整体剪掉的专家不计入
                continue
            counts[l, widths.index(int(w))] += 1
    return counts

def imbalance(loads):
    """论文定义 ΔΦ/Φ̄ = (max − min) / mean，以百分比返回。"""
    x = np.asarray(loads, dtype=np.float64)
    return (x.max() - x.min()) / x.mean() * 100.0

rows = []
for p, plan_path in [(0.30, "plans/qwen3_p30.json"), (0.50, "plans/qwen3_p50.json")]:
    full = width_counts_from_plan(plan_path, ACTIVE_WIDTHS)
    for L in (8, 24, 48):
        wc = full[:L]
        rec = {"p": p, "L": L}
        for method in ("greedy", "milp"):
            ts = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                res = solve_cross_layer_placement(wc, ACTIVE_WIDTHS, method=method)
                ts.append(time.perf_counter() - t0)
            rec[f"{method}_imbalance"] = imbalance(res["rank_weight_loads"].numpy())
            rec[f"{method}_time"] = float(np.median(ts))
        rows.append(rec)
        print(rec, flush=True)

json.dump(rows, open("results/placement_ablation.json", "w"), indent=2)
```

跑法：
```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p results logs
taskset -c 0-3 python scripts/run_placement_ablation.py 2>&1 | tee logs/placement_ablation.log
```
六行加起来应该是**分钟级**——m=4 的 MILP 只有 1152 个二元变量。

> [!bug] 有一个指标口径必须先对齐，否则填出来的数是错的
> 代码返回的 `relative_max_rank_weight_deviation` 定义是 **max|Φ_u − Φ̄| / Φ̄**（见 `_placement_objective` 及 greedy 返回值构造），而**表 6 caption 写的 ΔΦ/Φ̄、以及正文和 Algorithm 2 第 9 行定义的 ΔΦ = max_u(Φ_u) − min_u(Φ_u)** 是 max 减 min，不是 max 偏离均值。两者数值上差约 2 倍。
> **所以不要直接用 `relative_max_rank_weight_deviation` 填表**，要用返回的 `rank_weight_loads` 自己按 `(max−min)/mean` 算（上面脚本的 `imbalance()` 就是这么写的）。

> [!check] 你已决定：统一成 max−min，代码也改（2026-09-18）
> ==（你的意见）关于这个问题，我觉得可以改成 Max-min，代码里面也改成 Max-min，我知道代码里面现在可能是在求 deviation。你可以在 Markdown 里面提一下这个问题：要把代码里面的这个改成 Max-min。==
>
> **需要改代码的两个地方**（都在 `src/generate_mask/ep4_intplan.py`）：
>
> **① 优化目标函数 `_placement_objective`（第 843 行）**
> ```python
> # 现在
> def _placement_objective(rank_loads: np.ndarray) -> tuple[float, float]:
>     centered = rank_loads - rank_loads.mean()
>     return float(np.abs(centered).max()), float(np.dot(centered, centered))
>
> # 改成
> def _placement_objective(rank_loads: np.ndarray) -> tuple[float, float]:
>     spread = float(rank_loads.max() - rank_loads.min())       # ΔΦ = max − min
>     centered = rank_loads - rank_loads.mean()
>     return spread, float(np.dot(centered, centered))          # 第二项保留作 tie-break
> ```
> 第二个返回值（平方和）是局部搜索的次级比较键，用来在 ΔΦ 打平时选更均匀的解，**这个不要动**，它不影响报告的指标。
>
> **② greedy 返回值里的三个字段（第 936–939 行附近）**
> ```python
> # 现在
> mean = float(rank_loads_tensor.double().mean().item())
> max_deviation = float((rank_loads_tensor.double() - mean).abs().max().item())
> relative_deviation = max_deviation / mean if mean > 0.0 else 0.0
>
> # 改成
> loads = rank_loads_tensor.double()
> mean = float(loads.mean().item())
> max_deviation = float((loads.max() - loads.min()).item())     # ΔΦ = max − min
> relative_deviation = max_deviation / mean if mean > 0.0 else 0.0
> ```
> 改完之后 `max_rank_weight_deviation` 和 `relative_max_rank_weight_deviation` 这两个 key 的语义就和论文的 ΔΦ、ΔΦ/Φ̄ 一致了，脚本里可以直接用，不用再自己换算。
>
> [!warning] 改这两处会有两个连带影响，先确认再改
> **(a) `tolerance` 的含义变了。** `solve_cross_layer_placement(tolerance=0.01)` 是拿 `relative_deviation` 去比的（`tolerance_satisfied = relative_deviation <= tolerance`）。换成 max−min 之后同一个解的数值大约翻倍，原来 `tolerance=0.01` 能通过的，现在可能通不过。**如果生产 pipeline 在用这个 tolerance 做门控，要把默认值相应放宽**（经验上 0.01 → 0.02），否则剪枝流程可能开始报 tolerance 不满足。
> **(b) MILP 那一侧也要同步。** `_solve_placement_groups_milp`（第 680 行）的目标函数是独立建模的线性目标，不走 `_placement_objective`。如果它现在最小化的是"最大偏离均值"，就和 greedy 优化的目标不是同一个东西了，**两者的对比就不公平**。改之前先确认 MILP 的目标函数形式：如果它已经是 min(max−min) 就不用动，如果不是，要改成同一个目标。**这一点比前面两处更重要**——表 6 的整个论点是"greedy 逼近 MILP 的最优解"，前提是两者在优化同一个目标。
>
> 为保险起见，改完跑一次 `tests/test_ep4_intplan.py`，看有没有断言写死了旧口径的数值。

#### 建议的 m 扫描脚本（新增，跟表 6 的六行分开跑）

`scripts/run_m_sweep.py`，从 m=4 开始逐级加，**每级跑完立刻落盘**，MILP 带超时：

```python
import json, time, signal, torch, numpy as np
from src.generate_mask.ep4_intplan import solve_cross_layer_placement

MILP_TIMEOUT = 1800        # 秒。超时即记录为不可解，继续下一级
OUT = "results/m_sweep.json"

class Timeout(Exception): pass
def _alarm(signum, frame): raise Timeout()
signal.signal(signal.SIGALRM, _alarm)

def imbalance(loads):
    x = np.asarray(loads, dtype=np.float64)
    return (x.max() - x.min()) / x.mean() * 100.0

def run(width_counts, widths, method, timeout=None):
    """返回 (imbalance%, seconds) 或 (None, None) 表示超时。"""
    signal.alarm(int(timeout) if timeout else 0)
    try:
        t0 = time.perf_counter()
        res = solve_cross_layer_placement(width_counts, widths, method=method)
        dt = time.perf_counter() - t0
        return imbalance(res["rank_weight_loads"].numpy()), dt
    except Timeout:
        return None, None
    finally:
        signal.alarm(0)

rows = []
for m in (4, 5, 6, 7, 8, 12, 16):        # 从小到大，随时可以 Ctrl-C
    widths, wc = build_widths_and_counts(m)   # 见下方说明
    rec = {"m": m, "num_milp_binaries": wc.shape[0] * np.math.factorial(m)}
    g_imb, g_t = run(wc, widths, "greedy")
    rec |= {"greedy_imbalance": g_imb, "greedy_time": g_t}
    if m <= 7:
        mi, mt = run(wc, widths, "milp", timeout=MILP_TIMEOUT)
        rec |= {"milp_imbalance": mi, "milp_time": mt,
                "milp_status": "ok" if mi is not None else f"timeout>{MILP_TIMEOUT}s"}
    else:
        rec |= {"milp_status": "skipped (intractable)"}
    rows.append(rec)
    print(rec, flush=True)
    json.dump(rows, open(OUT, "w"), indent=2)   # 每级都落盘
```

`build_widths_and_counts(m)` 需要你按实际情况实现，两种做法：
- **做法 1（推荐）**：重新跑 `build_ep4_pruning_plan.py`，把宽度档数设成 m（要看脚本是否支持自定义档数，现在的预设是固定 5 档含 0）。这样每个 m 对应真实的剪枝方案。
- **做法 2（快速）**：固定一套剪枝结果，把宽度区间等分成 m 档重新量化。这样各 m 之间的对比更干净（同一个底层分布），但严格说不是"真实剪枝出来的 m 档"。**如果只是想展示求解器的扩展性，做法 2 完全够用**，而且更能说明问题，因为控制了变量。

跑 m≥8 前记得先改掉 `_solve_cross_layer_placement_greedy` 里的 `if num_widths <= 0 or num_widths > 7: raise ValueError` 守卫（改成比如 `> 32`）。

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p results logs
taskset -c 0-3 python scripts/run_m_sweep.py 2>&1 | tee logs/m_sweep.log
```


---------


## 实验 C 新版补充实验：greedy vs MILP 跨层重排求解器对比（填表 6）

> [!note] 这个实验在扫什么
> 跨层重排要解的问题是：每一层的专家被量化成若干**宽度档**（384/512/640/768），每个档在该层占用的权重显存是 `φ = 档宽 × 该档 expert 数`。要把这些档分配给 **m 个 expert-parallel rank（也就是 m 张卡）**，使各卡累计权重显存 `Φ_u` 尽量均衡，目标是 `ΔΦ = max_u Φ_u − min_u Φ_u` 最小。
>
> - **m 轴（主轴）**：扫的是 **EP 并行度**，即模型被切到几张卡上。m=4 是主实验配置（4×H20），m=8/16 对应更大规模部署。**问题规模随 m 增长**，问的是"并行度变大时，精确求解还撑得住吗"。
> - **L 轴（次轴）**：扫的是**参与重排的层数**。层数越多，可供跨层抵消的自由度越大。固定 m=4，从 48 层的 plan 里截取不同长度的连续窗口得到。
>
> 两个求解器解的是同一个问题：`greedy` 是论文 Algorithm 2，`milp` 是 `scipy.optimize.milp`（HiGHS）上的 assignment 建模。**二者只影响权重放在哪张卡上，不改动任何 expert 宽度、保留通道或路由决策，因此模型函数完全相同，精度一定一样**，差别只在显存均衡度。

### 一、实验设置（最终版）

| 项 | 设置 |
|---|---|
| 模型 / plan | Qwen/Qwen3-VL-30B-A3B-Instruct，48 层 × 128 experts，`d_model=2048`，宽度档 `(0, 384, 512, 640, 768)`（统计跳过 0 档） |
| 剪枝率 | p = 0.3 / 0.5，校准配置与主实验一致（`qwen3-mixed-512`） |
| 求解器 | `src/generate_mask/ep4_intplan.py` 的 `solve_cross_layer_placement(..., method="greedy"\|"milp")`；greedy 有 `full_bijection`（穷举 m! 双射）和 `pairwise_swap`（O(m²)）两个变体 |
| 指标 | `ΔΦ = max_u Φ_u − min_u Φ_u`（max−min 口径，代码已统一），报权重显存绝对值（MiB）+ 相对值 `ΔΦ/Φ̄`（%） |
| 单位换算 | φ 的单位是"通道×expert"，乘 `3·d_model·b`（gate/up/down 三矩阵）即为字节。bf16 下 **1 φ 单位 = 12 KiB**，最小非零偏差 128 单位 = **1.5 MiB** |
| MILP 时限 | **300 s**（此前用 3600s，实测 L=24 跑满一小时只换来 LB=0，纯浪费） |
| 多实例 | 从 48 层 plan 截取**不重叠**连续窗口：L=4 取 12 个、L=8 取 6、L=12 取 4、L=16 取 3、L=24 取 2、L=32/48 各 1。两个 p 合计 58 个实例 |
| 环境 | Linux，`taskset` 固定 4 核（24/26/28/30），`OMP/MKL_NUM_THREADS=4`，scipy 1.17.1，全程不改动 |
| 计时 | greedy 重复 5 次取中位数；MILP 每格在 60/180/300s 三个时限下各跑一次（求解器确定性，节点数完全一致，可当计时重复，CV ≤ 1.8%） |

> [!check] 有效性说明：两侧约束对称
> `fix_first_layer` 两个求解器一致——greedy 把第 0 层设 identity 并从第 1 层开始搜索，MILP 同样把第 0 层的 assignment 变量固定为 identity，调用侧都用默认 `True`。**所以测到的差距是真实的，不是约束不对称造成的。**

### 二、结论

**① m 轴上存在质量反超，这是支撑"默认用 greedy"的最强证据。**
m ≤ 8 时 MILP 能把偏差清零，但代价是 0.22 s → 53 s（greedy 的 2.5×10³ 倍）；**m = 12 时 MILP 烧满 300 s 只能打平；m = 16 时它在 300 s 内找到的解比 greedy 用 80 ms 得到的还差 2.5 倍**。也就是说超过 m≈12 后，精确求解不是"慢"，而是在任何现实预算下都被启发式压制。这个结论不依赖时限设得宽不宽。

**② 求解难度由"完美均衡是否可达"决定，不由问题规模决定。**
所有宽度档都是 128 的倍数，所以 ΔΦ 只能取 1.5 MiB 的整数倍；而 `min(max−min)` 的 LP 松弛下界恒为 0。两者叠加把实例劈成两类（判据在跑之前就能算出来）：

| 类别 | 判据 | MILP 的行为 |
|---|---|---|
| **均衡可达** | 总负载 ÷128 能被 m 整除 | 找到完美解时 UB=0 撞上 LB=0，gap 立即闭合，**秒级返回** |
| **均衡不可达** | 不能整除 | 下界仍是 0，必须搜完整棵树才能**证明 0 不可达**，300 s 全烧在这里 |

发生概率约 **1/m**（m=4 时理论 25%，实测 L≥8 的 34 个实例里 5 个 = 14.7%，二项检验 P(X≤5 \| n=34, p=0.25)=0.114，与 1/4 无显著差异）。**这不是病态实例，是四分之一的真实部署会遇到的正常情况。**

> [!warning] 可达实例不能删，只能分层报
> 可达实例恰好是 MILP 拿到 0、greedy 拿不到的那批，也就是 greedy 相对最差的样本。删掉它们等于专门剔除对我方法最不利的数据，而判据从 plan 里一行算术就能复现，审稿人一查就能还原。**做法是分层，并写明每层各有几个实例。**

**③ 不可达时精确求解基本拿不到可证明的最优。**
29 个不可达实例（L≥8）中，MILP 返回的解**全部**是 1.5 MiB——这是非零偏差的最小可能值。但其中只有 **11 个**（L=8 的 10 个 + L=12 的 1 个）在时限内**证明**了这是最优；其余 **18 个超时、下界仍停在 0**，只能说"300 s 内没找到更好的"，**不能称最优**。m=12 的 m 扫描里还有一个直接证据：60 s / 180 s / 300 s 三个时限返回的解完全相同，**后 240 秒没有换来任何改进**。

**④ 尺度感：整个争论的赌注是 3 MiB。**
实际部署档（m=4, L=48, p=0.5）下每张卡扛 **6.75 GiB** 专家权重，greedy 造成的不均衡是 **3.00 MiB**（0.043%），MILP 能把它清零。而且 **greedy 的 ΔΦ 在所有 m 上恒等于 3.00 MiB**，之前按百分比看到的"greedy 随 m 变差"是假象——百分比上升只因为每 rank 的 Φ̄ 从 6.75 GiB 缩到 1.69 GiB。

> [!bug] 三处旧说法已被实测证伪，不要再用
> **一、"m=8 已不可解"。** m=8 在 53 秒内证明了最优（ΔΦ=0）。断崖在 8 < m ≤ 12 之间，具体位置还没测。
> **二、"二元变量数是 `L × m!`"。** 日志里 `milp_binary_variables` = 768/1728/3072/6912/12288 = **48m²**，是 assignment 形式，规模 `L·m²`。`m!` 是 greedy 全双射邻域的大小，是另一个东西。（已核对：**论文 tex 第 524 行写法是对的**，那里的 `m!` 指的就是 greedy 邻域，不用改。）
> **三、"差距随 L 单调收敛（∞→13×→6×→2×）"。** 那是单实例的巧合。多实例下 greedy/MILP 比值是 1.07 → ∞ → 7 → 13 → 13.5 → 10 → 2，非单调，中间段最差。真正单调的只有绝对值。

### 三、结果

**表 C1：m 扫描（L=48，p=0.5，4 核，单实例；全部落在"可达"一侧）**

该 plan 的 φ_total ÷128 = 18432，能被 4/6/8/12/16 整除，**五个 m 的最优解都是 0**。但 m≥12 时 MILP 连这个已知存在的完美解都找不到：

| m | 每 rank Φ̄ | greedy ΔΦ | greedy 相对 | greedy 耗时 | MILP ΔΦ @300s | MILP 耗时 | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 6.75 GiB | 3.00 MiB | 0.043% | 6.2 ms | **0**（证明最优） | 0.22 s | MILP 赢 |
| 6 | 4.50 GiB | 3.00 MiB | 0.065% | 7.5 ms | **0**（证明最优） | 18.6 s | MILP 赢，慢 2.5×10³ 倍 |
| 8 | 3.38 GiB | 3.00 MiB | 0.087% | 21.5 ms | **0**（证明最优） | 53.0 s | MILP 赢，慢 2.5×10³ 倍 |
| 12 | 2.25 GiB | 3.00 MiB | 0.130% | 45.7 ms | 3.00 MiB（超时，LB=0） | 300 s | **打平**，白烧 6.6×10³ 倍时间 |
| 16 | 1.69 GiB | **3.00 MiB** | 0.174% | 80.1 ms | 7.50 MiB（超时，LB=0） | 300 s | **greedy 赢 2.5 倍** |

greedy 用 `pairwise_swap` 变体（m≥6 时）；m=4 上 `full_bijection` 与之结果相同（都是 3.00 MiB），耗时 17.3 ms vs 6.2 ms。

**表 C2：深度扫描按可达性分层（m=4，L≥8，greedy 用 pairwise swap，各格取中位数）**

| 类别 | L | 实例数 | 每 rank Φ̄ | greedy ΔΦ | greedy 相对 | MILP ΔΦ | MILP 耗时 | 超时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **可达** | 8 | 2 | 1.54 GiB | 36.75 MiB | 2.33% | **0** | 1.07 s | 0/2 |
| | 16 | 1 | 2.31 GiB | 21.00 MiB | 0.89% | **0** | 1.69 s | 0/1 |
| | 32 | 1 | 4.44 GiB | 9.00 MiB | 0.20% | **0** | 0.49 s | 0/1 |
| | 48 | 1 | 6.75 GiB | **3.00 MiB** | 0.043% | **0** | 0.22 s | 0/1 |
| **不可达** | 8 | 10 | 1.18 GiB | 37.50 MiB | 2.39% | 1.50 MiB | 15.59 s | 0/10 |
| | 12 | 8 | 2.04 GiB | 12.75 MiB | 0.63% | 1.50 MiB | **300.05 s** | 7/8 |
| | 16 | 5 | 3.06 GiB | 12.00 MiB | 0.54% | 1.50 MiB | **300.03 s** | 5/5 |
| | 24 | 4 | 4.06 GiB | 19.50 MiB | 0.42% | 1.50 MiB | **300.04 s** | 4/4 |
| | 32 | 1 | 6.26 GiB | 15.00 MiB | 0.23% | 1.50 MiB | **300.01 s** | 1/1 |
| | 48 | 1 | 9.45 GiB | **3.00 MiB** | 0.031% | 1.50 MiB | **300.05 s** | 1/1 |

greedy 耗时全在 1–7 ms（中位数 1.1 ms @L=8 到 6.3 ms @L=48），不随分层变化。

**分层之后"MILP 耗时随 L 恶化"这条就回来了**——不可达组 15.6 s → 300 s（7/8 超时）→ 全超时，干净单调；可达组则始终秒级、与 L 无关。此前看到的"L=48 反而比 L=12 快"纯粹是因为那一格恰好可达。

**L=4 单独说明（不进表）：** 24 个实例**全部不可达**——层数太少，即使整除也凑不出完美划分，**整除是必要不充分条件，只在 L≥8 上等价**。L=4 的 MILP 反而快（中位数 0.109 s），因为 4 层小到可以直接穷举；greedy 中位数 48.0 MiB，MILP 27.8 MiB，每 rank 仅 0.68 GiB。

### 四、还欠的数据

> [!todo] 按性价比排序
> **1. 补实例（最重要）。** 现在 L=32/48 每格只有 1 个实例，m 扫描更是全程 n=1——**m=16 的质量反超目前还是单实例结论**，不能当主结论写。三条扩充路径：
>
> | 做法 | 实例数变化 | 独立性 | 成本 |
> |---|---|---|---|
> | **滑动窗口 stride=4**（现在是不重叠切片） | 单个 p 从 29 → 55，两个 p 共 110 | 窗口高度重叠，**不独立**，只能缩小窗口选取的偶然性 | greedy 免费；MILP 需按下面第 2 条控制 |
> | **补剪枝率档** p ∈ {0.2, 0.4, 0.6} | ×2.5 | **独立**，且顺带回答"结论是否依赖剪枝率" | 每个 p 要重跑一次 build plan，校准分数可复用 |
> | **补模型**：Kimi-VL-A3B、InternVL3.5-30B-A3B（主表已有） | ×3 | **最独立**，且顺带支撑跨模型泛化 | plan 若已跑好则几乎免费 |
>
> 建议组合：**滑窗 stride=4 + 三个模型**，m 扫描至少要在 p=0.3 上再跑一轮把 n=1 变成 n=2。
>
> **2. MILP 时限能不能降到 60 s。** greedy 毫秒级，加多少实例都免费；瓶颈全在 MILP。有迹象说明 300 s 是浪费——m=12 在 60/180/300 s 返回完全相同的解，深度扫描所有超时格子也都停在 1.5 MiB。**但深度扫描没有 60 s 的对照数据**，建议先挑 5 个超时格子做 60 s vs 300 s 对照，确认无损后再降，这样同样预算能多跑 5 倍实例。（注意 m 扫描不能降：m=16 从 60 s 的 27 MiB 改进到 180 s 的 7.5 MiB。）
>
> **3. m = 9, 10, 11。** 定位断崖，现在只能说"在 8 和 12 之间"。≤15 min。
>
> **4. m = 5, 7。** 补齐原设计网格，图才是连续曲线而不是三个点。≤2 min。
>
> **5. m = 5/6/7 的 `greedy_full_bijection` 对照。** 论文第 524 行 "For larger m, the candidate strategies can be restricted to pairwise swaps … $O(m^2)$" 目前只有成本侧证据（m≥6 跑的就是 pairwise swap，ΔΦ 都是 3.0 MiB，与 m=4 全双射相同），缺的是同一 m 上两个变体的直接对照，用来说明"退化成 pairwise swap 没有损失质量"。≤5 min。（m=4 上已有 58 个实例的对照：49 平、4 胜、5 负，符号检验 p=1.0，无系统差异。）
