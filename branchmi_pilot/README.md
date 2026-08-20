# BranchMI counterfactual pilot

BranchMI 预调研实验。回答一个更早、更关键的问题：在不知道标准答案的在线分数中，answer-probe BranchMI 是否比局部 token entropy 更能预测“替换下一 token 会不会改变最终答案”。

代码会完整执行以下流程：

1. 为每道题生成一条原始、确定性的 CoT。
2. 在 CoT 中选若干 checkpoint；在每个位置重新计算下一 token 分布并取 top-2/3 候选。
3. 从每个候选 token 批量生成完整 continuation，构造 oracle criticality 标签。
4. 同时计算局部 entropy、varentropy、64-token short-lookahead JSD 和 answer-probe BranchMI。
5. 统计 AUROC、top-10% critical precision、probe/full-continuation 一致率和按题目聚类的 bootstrap 置信区间。
6. 自动生成 `continue` / `stop` 报告、CSV 和图表。

## 1. 实验定义

在原始回答生成 token 序列的第 `t` 个位置，保留位置之前的 prompt 和 reasoning prefix，重新计算：

- `entropy_nats`：完整 vocabulary 上的下一 token entropy，单位为 nat。
- `varentropy_nats2`：下一 token surprisal 的方差。
- `lookahead_js_mean_nats`：对 top-k 候选分别 greedy look ahead；在每个未来步比较各分支自己的 next-token 分布，计算加权 generalized JSD，再对有效步取平均。
- `branchmi_weighted_nats`：每个候选完成短 lookahead 后，另起一个不进入主路径的 user probe；每个分支采样多次短答案，形成经验分布 `q_j(A)`，再计算

  ```text
  H(sum_j w_j q_j) - sum_j w_j H(q_j)
  ```

  其中 `w_j` 是 top-k token 的原始概率在候选集合内重新归一化后的权重。

完整 continuation 使用相同的总回答 token 上限。最终答案先抽取，再用 `math-verify` 做符号等价聚类：

- `oracle_answer_change = true`：至少两个候选产生不等价的最终答案。
- `oracle_correctness_change = true`：至少一个候选正确、另一个错误。

默认主标签是 `oracle_answer_change`。这是有意的：答案变化不依赖 gold evaluator 的误差，而且与“这个位置是否值得保留多个分支”最直接。可在配置中把 `analysis.primary_label` 改成 `oracle_correctness_change`。

## 2. 环境安装

建议 Python 3.10–3.12、Linux 和 NVIDIA GPU。Qwen3-8B 的 BF16 权重本身约占 16 GB，批量分支和长 KV cache 还需要额外显存，实际建议 24 GB 或更多。CPU 和 Apple Silicon 可以运行 smoke test，但不适合 200 题正式实验。本机依赖已经安装在 Conda 环境 `LLM` 中。

```bash
conda activate LLM
cd branchmi_pilot
python -m pip install --upgrade pip
pip install -e .
```

若机器需要特定 CUDA 版本的 PyTorch，请先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装对应 wheel，再执行 `pip install -e .`。模型和 MATH-500 会由 Hugging Face 自动下载；需要使用自定义缓存目录时，正常设置 `HF_HOME` 即可。

安装开发工具并跑单元测试：

```bash
conda activate LLM
pip install -e '.[dev]'
pytest -q
ruff check src tests
```

## 3. 先跑 smoke test

smoke 配置使用 135M 模型、2 道本地题、每题 2 个 checkpoint。它只验证代码路径，不产生有意义的科研结论。

```bash
branchmi-pilot run --config configs/smoke.yaml
```

也可以不用安装后的命令入口：

```bash
python -m branchmi_pilot run --config configs/smoke.yaml
```

成功后会出现：

```text
outputs/smoke/
├── resolved_config.yaml
├── environment.json
├── problems.jsonl
├── checkpoints.jsonl
├── checkpoint_metrics.csv
├── summary.json
├── report.md
└── figures/
```

小样本可能只有单一 oracle 类别，此时 AUROC 显示为 `N/A`、决策为 `stop`，这是正常现象。

## 4. 正式跑 200 道 MATH-500

Qwen3-8B：

```bash
branchmi-pilot run --config configs/pilot_math500.yaml
```

DeepSeek-R1-Distill-Qwen-7B：

```bash
branchmi-pilot run --config configs/pilot_deepseek_7b.yaml
```

建议先用 5 道题估算显存和时间，并使用新的 run name，避免与正式结果混在一起：

```bash
branchmi-pilot run \
  --config configs/pilot_math500.yaml \
  --limit 5 \
  --run-name math500_qwen3_8b_dryrun
```

正式配置默认参数如下：

- 200 道随机但固定 seed 的 MATH-500；
- 每题最多 8 个 checkpoint；
- 每个位置 top-3 token；
- 64-token greedy lookahead；
- 每个分支 4 个 sampled answer probes；
- 原始回答与 counterfactual continuation 的统一总上限为 1536 tokens；
- oracle continuation 为 greedy，以避免 oracle 标签被额外采样噪声污染；
- probe 使用 `temperature=0.7, top_p=0.95`，用于估计 `q_j(A)`。

正式实验的主要计算量来自：

```text
题数 × checkpoint 数 × top-k 分支 × 剩余完整 continuation 长度
```

因此第一次运行前应先做 5 题 dry run。代码已经把同一 checkpoint 的 top-k lookahead、probe 和 oracle continuation 分别组成 batch，但不会隐藏真实生成成本。

## 5. 中断与恢复

`problems.jsonl` 和 `checkpoints.jsonl` 都是完成一项立即 append、flush 和 fsync。进程被杀死、OOM 或机器重启后，执行完全相同的命令即可从未完成的 checkpoint 继续：

```bash
branchmi-pilot run --config configs/pilot_math500.yaml
```

为防止把不同实验混入同一目录，首次运行会将配置冻结到 `resolved_config.yaml`。之后若同一个 `run_name` 的配置有任何变化，程序会拒绝恢复。修改参数时请同时修改 `run_name`。

不要并行启动两个写入同一 `run_name` 的进程；JSONL writer 不提供跨进程锁。

## 6. 单独重跑分析

分析不需要加载模型：

```bash
branchmi-pilot analyze --run-dir outputs/math500_qwen3_8b_pilot
```

这会从 `checkpoints.jsonl` 去重并重建 CSV、`summary.json`、Markdown 报告和图表。修改停止阈值或 bootstrap 次数时，应编辑该运行目录里的 `resolved_config.yaml`，或者把运行目录复制为新的分析版本后再改；原始 checkpoint 数据不会被改写。

## 7. 如何读结果

首先打开 `outputs/<run_name>/report.md`。核心比较是：

```text
AUROC(BranchMI) - AUROC(entropy)
top-10% precision(BranchMI) / top-10% precision(entropy) - 1
probe mode answer 与 full continuation answer 的逐分支一致率
```

配置按预研标准使用：

- AUROC 增益至少 `0.08`；
- top-10% critical precision 相对提升至少 `15%`；
- probe/full-continuation 一致率至少 `70%`。

前两条直接来自止损线。第三条原始描述没有给数值，代码将“能够近似”操作化为 70% 的逐分支答案一致率，可在 `analysis.min_probe_oracle_agreement` 中修改。只有三条都达到时，报告才给出 `continue`。

置信区间按 problem bootstrap：一次重采样会带走该题的全部 checkpoint，避免把同题位置错误地当成独立样本。`summary.json` 还记录 AUROC 增益和 precision 增益的配对 bootstrap 区间。

图表包括：

- `roc_curves.png`：各在线分数预测 oracle criticality 的 ROC；
- `auroc_comparison.png`：AUROC 柱状比较；
- `entropy_vs_branchmi.png`：局部 entropy 与未来答案影响的散点图。

## 8. 关键配置项

### 数据

```yaml
dataset:
  kind: hf                    # hf 或 jsonl
  path: HuggingFaceH4/MATH-500
  split: test
  question_field: problem
  answer_field: answer
  limit: 200
  shuffle: true
  seed: 2026
```

本地 JSONL 每行至少需要问题字段；没有 gold 时可以省略答案字段，但 `oracle_correctness_change` 将无法使用。

### checkpoint

```yaml
checkpoints:
  strategy: hybrid            # uniform / stride / markers / hybrid
  max_per_problem: 8
  stride: 96
  min_generated_tokens: 32
  tail_exclusion_tokens: 32
  markers: ["\n", Wait, Alternatively, Therefore, Thus]
```

`generated_position=t` 的含义是保留 `generated_ids[:t]`，替换原来的 `generated_ids[t]`。`hybrid` 合并均匀覆盖、固定间隔和 marker 后再确定性下采样到上限。

如果原始回答太短，不满足前缀和尾部排除条件，该题可能没有 checkpoint；这不会伪造样本。

### 生成

```yaml
generation:
  baseline_max_new_tokens: 1536
  oracle_max_total_tokens: 1536
  candidate_top_k: 3
  lookahead_tokens: 64
  probe_samples: 4
  probe_max_new_tokens: 48
  probe_temperature: 0.7
  probe_top_p: 0.95
```

`oracle_max_total_tokens` 是 assistant 整条回答的统一上限，不是每个 checkpoint 之后再给 1536 tokens。这样越晚的 checkpoint 不会凭空获得更大预算。

### Qwen3 thinking 模式

Qwen3 主路径默认：

```yaml
chat_template_kwargs:
  enable_thinking: true
```

answer probe 默认关闭新的 thinking block：

```yaml
probe_chat_template_kwargs:
  enable_thinking: false
```

probe 是在 chat history 中加入一条临时 user message后生成，生成内容不会回写主 reasoning path。不同模型不接受这些 template 参数时，像 DeepSeek 示例一样设成 `{}`。

## 9. 输出数据结构

`problems.jsonl` 每题一行，保存：

- 问题、gold、原始生成 token IDs；
- 原始模型回答、抽取答案和正确性；
- prompt / response token 数。

`checkpoints.jsonl` 每个位置一行，保存：

- checkpoint 位置和被替换的原 token；
- entropy、varentropy、lookahead JSD、weighted/uniform BranchMI；
- 两种 oracle criticality 标签；
- 每个分支的 token 概率、截断后权重、probe samples、完整答案、正确性和 token 数；
- `save_full_text=true` 时的 lookahead、probe completion 和完整 counterfactual response。

为了便于审计，原始 token 概率和 top-k 内重新归一化的 `branch_weight` 会同时保留。BranchMI 主结果用后者；`branchmi_uniform_nats` 是消融。

## 10. 已知限制与下一步

- 这是 token-level counterfactual pilot，不是最终论文中的 32–64 token next-thought proposal。若 pilot 成功，下一步再把候选单元升级成 step-level branch。
- top-k token 只覆盖截断候选集合；非常低概率但关键的 token 不在本实验 oracle 中。
- lookahead JSD 比较的是各分支沿自身轨迹的 future next-token distributions。不同轨迹的“第 20 步”未必语义对齐，所以它是诊断 baseline，不应解释成 mutual information。
- 经验 answer distribution 的分辨率受 `probe_samples` 限制。增大它会更稳定，也会线性增加 probe 成本。
- `math-verify` 已优先做符号等价；解析失败时退回保守的规范化字符串比较。建议在论文实验前人工审计 critical/non-critical 各 50 个案例。
- 单次模型、prompt 和 decoding policy 的 negative result 不能证明 BranchMI 概念无效，但三天止损实验应严格按预注册阈值执行，避免后验调参。

实现使用 Hugging Face 官方的 [`generate`](https://huggingface.co/docs/transformers/main_classes/text_generation) / generation scores 接口和 [`load_dataset`](https://huggingface.co/docs/datasets/loading)；数学答案等价判定使用 [Hugging Face Math-Verify](https://github.com/huggingface/Math-Verify)。
