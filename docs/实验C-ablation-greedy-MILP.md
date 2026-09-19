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
