# Qwen3-VL 30B MoE 剪枝接入 vLLM 四卡 EP 计划

## 1. 目标与边界

本分支用于将 MAES 对 `Qwen/Qwen3-VL-30B-A3B-Instruct` 的结构化专家通道剪枝接入 vLLM，并首先保证以下配置的数值正确性：

- BF16；
- 4 张 GPU；
- Expert Parallel；
- eager 模式；
- 继续使用 vLLM 原有的 Fused MoE / grouped GEMM kernel；
- 不开发新的极限优化 GEMM kernel；
- 全局 router、原始 top-k 选择及原始 top-k routing weight 计算保持不变。

实现必须使用独立的 vLLM 入口脚本，不修改现有评测和 sweep 脚本。计划中的脚本名为：

```text
scripts/run_qwen3_vl_vllm_ep.sh
```

该入口最终应支持 `baseline`、`pruned` 和 `compare` 三种模式。

## 2. Git 与现有工作树状态

2026-09-09 已执行 `git fetch --prune origin` 并确认：

- 开发基线为 `dev`；
- `HEAD` 与 `origin/dev` 均为 `61f1a090d0ce506dc5b5cf08101bcba03da0744d`；
- `HEAD...origin/dev` 的 ahead/behind 为 `0/0`；
- `origin/main` 虽有日期更晚的 README 提交，但相对 `dev` 缺少大量开发提交，不作为本工作的基线；
- 新分支名称统一使用小写 `vllm`。

创建 `vllm` 分支时，以下用户已有的未提交修改原样保留，不覆盖、不回退，也不自动纳入 vLLM 实现提交：

```text
download_hf_benchmarks.py
scripts/prune_and_eval_kimi_gqa.py
scripts/run_prune_eval_kimi_gqa.sh
scripts/sweep_video_prune_eval.sh
src/base/models/qwen3.py
```

其中 `scripts/prune_and_eval_kimi_gqa.py` 同时有 staged 和 unstaged 修改。后续提交必须按文件精确暂存，不能使用会混入这些修改的宽泛暂存命令。

## 3. 已核实的 Qwen3-VL MoE 结构

离线模型缓存：

```text
/home/data/dyf/hf_cache/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct
```

模型 revision：

```text
9c4b90e1e4ba969fd3b5378b57d966d725f1b86c
```

`config.json` 的相关字段为：

| 字段 | 数值 | 含义 |
|---|---:|---|
| `hidden_size` | 2048 | token hidden dimension |
| `intermediate_size` | 6144 | 普通 dense MLP 配置，不是 routed expert 宽度 |
| `moe_intermediate_size` | 768 | routed expert 的真实中间层宽度 |
| `num_hidden_layers` | 48 | MoE 层数 |
| `num_experts` | 128 | 每层全局 routed experts |
| `num_experts_per_tok` | 8 | router top-k |
| `decoder_sparse_step` | 1 | 每层均为 sparse MoE |
| `mlp_only_layers` | `[]` | 没有额外 dense-only decoder layer |

safetensors 元数据进一步确认第 0 层和第 47 层均为：

```text
gate_up_proj: (128, 2048, 1536)  # 1536 = 2 * 768
down_proj:    (128, 768, 2048)
gate.weight:  (128, 2048)
```

结论：每个 routed expert 的原始 intermediate dimension 就是 768，没有比 768 更宽的 routed expert。`intermediate_size=6144` 不能用于专家剪枝宽度判断。

## 4. 专家宽度与删除语义

活动专家只允许四档宽度：

```text
768, 512, 384, 256
```

完整的专家状态集合为：

```text
0, 256, 384, 512, 768
```

其中宽度 0 不是第五种计算宽度，而是“删除该专家”。不能为了满足最小 kernel shape 而把它强制恢复为 256。

每层保存 `expert_width[layer, global_expert_id]`。宽度为 0 时：

- 全局 router 的输出维度仍为 128；
- router logits 不裁切；
- 仍按原模型执行 softmax、top-8 和原有 top-k normalization；
- top-k 完成后检查每个选中专家的 `expert_width`；
- 若宽度为 0，将该 token-expert pair 改成 `expert_id=-1` 且 `routing_weight=0`；
- 不对剩余 routing weights 再归一化，以保持“被删除专家贡献为零”的结构化剪枝语义；
- 若一个 token 的 8 个专家全部被删除，则该层 routed MoE 对该 token 的输出为零。

使用 `-1` 作为唯一无效 ID。暂不使用 `num_experts`（128）作为 sentinel，因为部分 `expert_map[id]` 路径会把它当作数组下标，存在越界风险。

伪代码如下：

```python
router_logits = gate(hidden_states)                 # [T, 128]
topk_weights, topk_ids = original_topk(router_logits, k=8)
active = expert_width[layer_id, topk_ids] > 0
topk_weights = topk_weights.masked_fill(~active, 0)
topk_ids = topk_ids.masked_fill(~active, -1)
```

## 5. vLLM 无效 expert ID 可行性

本次只为可行性检查读取了 vLLM 上游源码，检查点为：

```text
c7e9816c6ab0731165a134fd0a9defed6ab1d748
```

该 commit 只作为可行性证据，不代表最终依赖版本。正式实现前仍需选定并固定一个可构建、支持 Qwen3-VL-MoE 的 vLLM release 或 commit，baseline 与 pruned 必须使用完全相同的版本。

源码检查结论：`-1` 无效 ID 可以实现，并且 vLLM 已有部分底层语义支持：

- `count_expert_num_tokens` 明确要求 signed ID，并声明 `-1` 代表 invalid top-k ID；
- token count kernel 不会把 `-1` 计入任何本地专家；
- `moe_align_block_size(..., ignore_invalid_experts=True)` 会让无效 pair 不参与计数、排序和 padding，因此不会产生对应 GEMM block；
- pad-aware activation kernel 会跳过 `topk_ids == -1`；
- MoE reduce 路径已有“invalid slots excluded”测试；
- DeepEP 接收侧本身也存在 `-1` padding slot 语义。

因此无需修改 BF16 grouped GEMM 数学 kernel。需要补齐和验证的是 router 后 mask、placement、all-to-all dispatch、local ID mapping 和 combine 的端到端一致性。

## 6. vLLM 默认假设与拟修改边界

当前上游实现仍包含不适用于本项目的默认假设：

- `ExpertMapManager.determine_expert_map` 只支持均匀 linear/round-robin placement；
- Qwen3 MoE block 直接使用所有 rank 相同的 `config.moe_intermediate_size`；
- `n_local_physical_experts` 默认按全局专家数除以 EP size；
- 部分 DeepEP/TRT-LLM/FlashInfer 路径使用 contiguous rank expert offset；
- EPLB 和 elastic EP 的部分状态要求各 rank 相同的本地专家数量；
- 不同 kernel backend 对 arbitrary `expert_map` 和 invalid ID 的支持程度不同。

第一阶段固定以下边界：

- BF16 unquantized；
- `enable_eplb=False`；
- `enforce_eager=True`；
- 不启用 elastic EP；
- 不启用依赖 contiguous/equal placement 的 TRT-LLM MoE backend；
- 优先选择明确支持 `expert_map` 和 `-1` 的 vLLM Triton FusedMoE 路径。

实现放在 MAES 仓库的 out-of-tree 集成模块中。优先使用自定义类和窄范围 adapter；只有选定 vLLM 版本无法注入必要映射时才使用 monkey patch。Monkey patch 必须：

- 检查 vLLM 精确版本或 commit；
- 检查目标函数签名；
- 不直接修改 conda 环境中的 site-packages 文件；
- 版本不匹配时立即报错，不能静默继续；
- 配套单元测试覆盖被 patch 的接口。

## 7. Pruning/placement manifest

新增一个独立的 vLLM pruning plan compiler，读取 MAES 的 `scores.pt`、mask 结果和模型 config，生成版本化 manifest。manifest 至少包含：

```text
schema_version
model_id
model_revision
scores_path / scores hash
allowed_active_widths = [768, 512, 384, 256]
expert_width[layer, global_expert_id]
channel_indices[layer, global_expert_id]
expert_to_rank[layer, global_expert_id]
expert_to_local_id[layer, global_expert_id]
local_to_global[layer, rank, local_expert_id]
routing_frequency[layer, global_expert_id]
estimated_rank_load[layer, rank]
```

映射约束：

- 宽度 768、512、384、256 分别一对一映射到四个 GPU rank；具体 rank permutation 固定写入 manifest；
- 同一 rank 内所有活动专家宽度相同；
- 宽度为 0 的专家满足 `expert_to_rank=-1`、`expert_to_local_id=-1`；
- 每层各 rank 的 `num_local_experts` 可以不同；
- 每层活动专家的 global/local mapping 必须是双射；
- 每个活动 global expert 只能属于一个 rank；
- 删除专家不加载任何 gate/up/down 权重。

如果某层某个宽度档没有活动专家，对应 rank 仍必须参加 collective，但跳过本地 FusedMoE 计算。不能为了让 tensor 非空而加入会影响结果的假专家。

## 8. 结合路由频率的宽度分配

校准产物中已有逐层 expert routing 统计，例如 `token_count_text`、`token_count_visual`、`usage` 和 `router`。plan compiler 将明确选择一种 routing frequency 定义，并把选择写入 manifest。

默认候选为：

```text
p_e = alpha * p_text_e + (1 - alpha) * p_visual_e
```

若产物保存了真实 text/visual token 总量，则由真实比例确定 `alpha`；若只有分模态归一化频率，则必须显式传入 `alpha`，不能暗中假设 0.5。

宽度和专家数量分配在满足总参数/通道预算及 channel importance 的前提下，最小化四卡预计计算负载差异：

```text
load(layer, rank) = sum(p_e * d_e)
```

其中 `d_e=0` 的专家负载为 0。优化结果同时报告：

- 每层各宽度专家数量；
- 每层删除专家数量；
- 每 rank 的 `sum(p_e * d_e)`；
- 最大/最小 rank load ratio；
- 相对原始 mask 预算的偏差。

通道选择仍按 MAES channel importance 完成；宽度离散化不能只截断 mask 的前若干布尔位置，而应按对应 score 排序选择准确的 `d_e` 个通道。

## 9. 自定义 Qwen3-VL vLLM 模型

计划新增独立模块，具体文件名在选定 vLLM API 后确定，职责划分如下：

- 模型注册：注册自定义 Qwen3-VL-MoE architecture，不修改原 checkpoint；
- plan loader：读取并严格验证 manifest、模型 revision 和 tensor shape；
- sparse MoE block：保留完整 replicated gate，在 top-k 后应用 deleted-expert mask；
- per-layer expert map：向每层注入各自的 global-to-rank/global-to-local mapping；
- local FusedMoE：每个 rank 使用自己的 `num_local_experts` 和固定 width；
- weight loader：只加载本 rank 活动专家，并按 channel indices 切片；
- dispatch/combine adapter：完成 global ID、owner rank、local ID 的转换和反向 combine。

自定义 architecture 通过 vLLM `ModelRegistry` 在创建 engine 前注册。运行脚本使用 config override 指向自定义 architecture，原始 Hugging Face config 和缓存保持只读。

## 10. 本地专家权重加载

原始 checkpoint 使用 fused 3D tensor：

```text
gate_up_proj: [128, 2048, 1536]
down_proj:    [128, 768, 2048]
```

每个 rank 的 loader：

1. 根据逐层 `local_to_global` 找出本 rank 活动专家；
2. 根据每个专家的 `channel_indices` 从 gate、up、down 三部分取相同通道；
3. 转换为所选 vLLM BF16 FusedMoE backend 所需的 w13/w2 布局；
4. pack 成该 rank 的统一形状；
5. 不实例化或复制其他 rank 的专家权重；
6. router gate 权重仍完整加载并复制到所有 EP ranks。

loader 测试必须从首层、中间层和末层抽取专家，将 packed tensor 与原 safetensors 对应切片逐元素比较。

## 11. Dispatch、local GEMM 与 combine

目标数据流：

```text
global router logits [T, 128]
  -> original softmax/top-8/normalization
  -> deleted expert mask: id=-1, weight=0
  -> global expert id -> owner rank
  -> all-to-all dispatch（invalid pair 不进入 send counts/prefix sums）
  -> global expert id -> destination local expert id
  -> rank-local FusedMoE/grouped GEMM
  -> reverse all-to-all
  -> 按原 token、top-k slot 和 routing weight combine
```

实现顺序：

1. 先验证选定 vLLM 版本的 DeepEP backend 是否同时支持 `-1`、任意逐层 placement 和不等 `num_local_experts`；
2. 若支持，复用 DeepEP dispatch/combine，只替换 mapping metadata；
3. 若 DeepEP 仍依赖等量连续 partition，第一版实现基于 PyTorch/NCCL `all_to_all_single` 的正确性 dispatcher；
4. 无论使用哪种 dispatcher，本地专家计算均复用 vLLM 原有 BF16 FusedMoE/grouped GEMM；
5. 不通过把所有 hidden states 广播到所有 rank 来伪装 EP 性能收益。

因此该方案在工程上可实现。主要风险不在 `-1` 本身，而在选定 vLLM/DeepEP 版本对 arbitrary placement 和 unequal local expert count 的 metadata 假设。

## 12. 环境计划

当前 `maes` conda 环境没有安装 `vllm` 或 `lmms_eval`。计划新建小写环境：

```text
vllm
```

环境要求：

- 固定 Python、PyTorch、CUDA、vLLM 和 lmms-eval 版本；
- 记录完整 `conda list` / `pip freeze`；
- 使用统一缓存：

```text
HF_HOME=/home/data/dyf/hf_cache
HF_HUB_CACHE=/home/data/dyf/hf_cache/hub
HF_DATASETS_CACHE=/home/data/dyf/hf_cache/datasets
```

- 模型和数据必须以 offline 模式加载，不重新下载；
- baseline 与 pruned 在同一个环境、同一个 vLLM commit 中运行。

## 13. 数值正确性验证

按以下顺序推进，前一阶段未通过不得进入性能测试：

1. Manifest 单元测试：宽度集合、删除语义、mapping 双射、预算和负载统计；
2. Weight loader 单元测试：本地 expert slicing 与 checkpoint 精确一致；
3. 单层 synthetic 测试：包含部分和全部 top-k pair 被删除的 token；
4. 四卡 dispatch 测试：检查 send counts 中没有 deleted pair，local ID 范围正确；
5. BF16 MoE reference：与 PyTorch 逐专家实现比较输出误差；
6. Router 一致性：原始与自定义路径的 router logits、top-k IDs、mask 前 weights 一致；
7. 端到端图文 smoke test：固定输入、greedy decode，并比较关键层 hidden states/logits；
8. 多 batch size、输入长度、输出长度和并发度回归。

日志必须至少记录：

- 每层各 rank 活动专家数和宽度；
- 每层删除专家数；
- invalid pair 数量和比例；
- 每 rank 实际接收的 token-expert pair 数；
- mapping/manifest/model revision；
- BF16 reference 的最大绝对误差和相对误差。

## 14. 公平性能评测

原始模型使用 vLLM stock Qwen3-VL-MoE 路径，剪枝模型使用自定义路径。两者固定：

- 同一 vLLM commit；
- 4 卡并行配置；
- BF16；
- eager 模式；
- 相同 batch size；
- 相同输入长度和输出长度；
- 相同并发度；
- 相同 prompt/media；
- 相同 scheduler、`max_num_batched_tokens`、KV cache 和显存利用率参数；
- 相同 warmup 次数和正式重复次数。

指标至少包括：

- 端到端 latency；
- TTFT；
- TPOT；
- request throughput；
- input/output/total token throughput；
- 每 rank peak allocated/reserved GPU memory；
- 模型加载后的静态显存；
- 各 rank 实际 token-expert pair 数和负载偏差。

## 15. Video-MMMU 状态

2026-09-09 已一次性检查：

```text
路径: /home/data/dyf/hf_cache/datasets/VideoMMMU
MP4 数量: 301
空文件: 0
总大小: 13,821,999,601 bytes (12.873 GiB)
```

数据已经解压，不再在每次运行前执行全目录扫描，也不再运行 `prepare-vmmmu`。实际评测仍应在样本日志中记录媒体路径和实际采样帧数；非 Adaptation 样本若指定媒体缺失，应立即报错，不能静默退化为静态图。

## 16. 预期新增文件与不修改项

预期新增：

```text
src/vllm_integration/
tests/vllm_integration/
scripts/run_qwen3_vl_vllm_ep.sh
docs/vllm_qwen3_vl_ep_pruning_plan.md
```

具体 Python 文件将在 vLLM 版本固定后按实际 API 划分。现有 `scripts/prune_and_eval_kimi_gqa.py`、`scripts/run_prune_eval_kimi_gqa.sh` 和 sweep 脚本不作为 vLLM 入口，不因本工作改写。

## 17. 完成标准

第一阶段完成必须同时满足：

- 每个活动专家宽度属于 `{256,384,512,768}`；
- 宽度 0 专家不加载权重、不参与 dispatch、不参与 GEMM；
- 全局 router 和 top-k mask 前结果与原模型一致；
- 四卡各 rank 可有不同 `num_local_experts` 和不同固定 intermediate dimension；
- 每 rank 内活动专家形状一致并使用原有 vLLM FusedMoE/grouped GEMM；
- BF16 eager 四卡输出通过 reference 数值测试；
- baseline/pruned 公平基准能够复现并输出完整配置、速度和显存报告；
- 不重新下载已有模型或数据；
- 不覆盖本分支建立前已有的工作树修改。
