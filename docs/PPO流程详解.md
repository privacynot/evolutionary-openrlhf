# MASDiff_change 中 PPO 流程详解

本文说明 `Masdiff_change` 工程中 PPO 是怎样被调用和实现的、每一次 PPO 的输入输出分别是什么，以及 `outputs/ppo_io_snapshot/ppo_io_snapshot.json` 中每个字段应该如何理解。

重点结论先放在前面：本项目里的 `response` 不是 step 级训练单位，也不是逐 token 单独生成的独立样本。Qwen 一次生成的是一整段 response；代码把这整段 response 编码成 `sequences`，用 `action_mask` 标出哪些 token 属于生成部分，再把扩散模型生成的 reward 序列对齐到 token 维度上做 PPO 更新。因此，样本单位是“整条 response 序列”，优化计算发生在 token 维度。

## 相关文件

PPO 主流程主要涉及这些文件：

- `src/pipeline/steps_LLM.py`：定义 4.1 到 4.5，以及 5.3 中第二次 PPO 的步骤函数。
- `src/llm/qwen_module.py`：真实 Qwen 的 `generate()` 和 `train_ppo()` 实现。
- `src/diffusion/base.py`：扩散模型根据 `Tau / hidden_states` 生成奖励序列 `R`。
- `src/pipeline/runner_LLM_new.py`：完整 MASDiff 进化式流程，包含初始 PPO 和变异后的第二次 PPO。
- `masdiff_change/runner_no_evolution.py`：no-evolution 入口，主要运行初始个体构建与训练。
- `export_ppo_io_snapshot.py`：用于导出 PPO 输入输出快照的脚本。
- `outputs/ppo_io_snapshot/ppo_io_snapshot.json`：已经导出的第一次 PPO 和第二次 PPO 输入输出示例。

## 总体流程

在 LLM 版本中，PPO 不是单独孤立运行的，而是夹在“Qwen 生成 response”和“扩散模型生成 reward”之间。

整体关系如下：

```text
prompt / label
    ↓
Qwen.generate()
    ↓
response, sequences, action_mask, hidden_states(Tau)
    ↓
DiffusionModel.generate_reward(Tau)
    ↓
reward sequence R
    ↓
build_ppo_data(response, sequences, action_mask, R)
    ↓
Qwen.train_ppo(training_data)
    ↓
更新后的 Qwen actor / critic 参数
```

其中：

- `prompt`：输入给 Qwen 的问题或任务描述。
- `label`：期望答案或参考反馈，用于后续评估，不直接作为 PPO loss 的监督标签。
- `response`：Qwen 一次生成的完整文本回答。
- `sequences`：完整 token id 序列，通常包含 prompt token 和 response token。
- `action_mask`：布尔 mask，用来标记哪些 token 属于生成动作，PPO loss 只应该作用在生成部分。
- `hidden_states` / `Tau`：Qwen 生成过程中的隐藏状态序列，是扩散模型的条件输入。
- `reward` / `R`：扩散模型输出的 reward 序列，后续会和 token 位置对齐。
- `training_data`：真正传给 `train_ppo()` 的 PPO 训练数据。

## 第一次 PPO：初始个体训练

第一次 PPO 发生在初始 population 构建阶段。对应代码路径是：

```text
step_4_1_generate_and_collect()
step_4_2_generate_prm_reward()
step_4_3_build_ppo_data()
step_4_4_train_qwen_ppo()
```

### 4.1 生成 response 并收集 PPO 数据

函数：`step_4_1_generate_and_collect()`

输入：

```text
qwen_module
prompts
labels
num_samples
model_index
```

内部会调用：

```python
qwen_module.generate(
    batch_prompts,
    return_hidden_states=True,
    model_index=model_index,
)
```

真实 Qwen 的 `generate()` 返回：

```text
responses
hidden_states
sequences
action_mask
logits_summary 可选
```

这一阶段的输出是：

```text
ppo_data_list
final_hidden_states
all_logits
```

其中 `ppo_data_list` 的每个样本大致是：

```json
{
  "prompt": "...",
  "label": "...",
  "response": "...",
  "sequences": [token_id, token_id, ...],
  "action_mask": [true, true, false, ...],
  "logits": null
}
```

注意：这里的 `response` 是整段文本，不是 step 列表。即使文本中包含 `Step 1`、`Step 2`，代码也没有把它拆成多个 step 样本。

### 4.2 扩散模型生成 reward

函数：`step_4_2_generate_prm_reward()`

输入：

```text
diffusion
hidden_states / Tau
ppo_data
step_tag_ids 可选
```

核心调用：

```python
raw_rewards_seq = diffusion.generate_reward(hidden_states)
```

输出：

```text
rewards
```

如果没有配置 `step_tag_ids`，`rewards` 是完整 token 序列长度上的 reward。形状通常类似：

```text
[batch_size, sequence_length]
```

如果配置了 `step_tag_ids`，代码会只保留 step tag 对应 token 位置上的 reward，其余位置置零。但默认情况下，本项目走的是整段序列 reward。

### 4.3 合并 PPO 数据和 reward

函数：`step_4_3_build_ppo_data()`

输入：

```text
ppo_data
rewards
```

它做的事情很直接：

```python
item["reward"] = rewards[i]
```

合并之后，每个 PPO 样本变成：

```json
{
  "prompt": "...",
  "label": "...",
  "response": "...",
  "sequences": [token_id, token_id, ...],
  "action_mask": [true, true, ...],
  "reward": [0.49, 0.33, 0.40, ...]
}
```

这就是 `train_ppo()` 的直接输入。

### 4.4 调用 PPO 训练

函数：`step_4_4_train_qwen_ppo()`

输入：

```text
current_qwen
training_data
model_index
```

核心调用：

```python
current_qwen.train_ppo(training_data, model_index=model_index)
```

输出：

```text
训练后的 current_qwen
```

这里不会返回新的 response。它返回的是已经被 PPO 更新过的 Qwen 模型对象。之后如果要得到新的 response，需要再次调用 `generate()`。

## 第二次 PPO：变异个体训练

第二次 PPO 出现在完整 MASDiff 进化流程的 step 5.3 中。它包含两个 PPO 更新点：

```text
5.3.1  truncated diffusion mutate
5.3.2  build PPO data with mutated rewards
5.3.3  train temp_qwen with mutated rewards
5.3.4  temp_qwen generates new response and Tau
5.3.5  diffusion generates standard rewards for new Tau
5.3.6  build PPO data again
5.3.7  train temp_qwen again
```

用户通常说的“第二次 PPO”，主要指 step 5.3 中对临时模型 `temp_qwen` 的 PPO 训练过程。它和第一次 PPO 的核心数据结构完全一样，只是数据来源不同。

### 5.3.1 生成变异 reward

函数：`step_5_3_1_truncated_diffusion_mutate()`

输入：

```text
diffusion
elite.tau
add_noise_steps
denoise_steps
```

输出：

```text
mutated_rewards
```

这一步不会生成新的 response，它只基于精英个体已有的 `tau` 生成一组变异 reward。

### 5.3.2 用旧 PPO 数据和变异 reward 构造训练数据

函数：`step_5_3_2_build_ppo_data()`

它实际复用的是 `step_4_3_build_ppo_data()`。

输入：

```text
old_ppo_data
mutated_rewards
```

输出：

```text
mutant_training_data
```

这里的 response、sequence、action_mask 来自旧精英个体，reward 换成了变异 reward。

### 5.3.3 第一次训练 temp_qwen

函数：`step_5_3_3_train_temp_qwen()`

输入：

```text
temp_qwen / qwen_module_template
mutant_training_data
model_index
```

核心调用：

```python
qwen_module_template.train_ppo(training_data, model_index=model_index)
```

输出：

```text
训练后的 temp_qwen
```

### 5.3.4 重新生成 response 和 Tau

函数：`step_5_3_4_generate_and_collect()`

它复用 `step_4_1_generate_and_collect()`。

输入：

```text
temp_qwen
prompts
labels
model_index
```

输出：

```text
new_ppo_data
tau_new
logits_new
```

这一步才会生成新的 response。

### 5.3.5 生成标准 reward

函数：`step_5_3_5_generate_standard_reward()`

它复用 `step_4_2_generate_prm_reward()`。

输入：

```text
diffusion
tau_new
new_ppo_data
```

输出：

```text
standard_rewards
```

### 5.3.6 再次构造 PPO 数据

函数：`step_5_3_6_build_ppo_data()`

输入：

```text
new_ppo_data
standard_rewards
```

输出：

```text
align_training_data
```

这时的 response、sequence、action_mask 来自 step 5.3.4 新生成的结果，reward 来自 step 5.3.5。

### 5.3.7 第二次训练 temp_qwen

函数：`step_5_3_7_train_temp_qwen()`

输入：

```text
temp_qwen
align_training_data
model_index
```

核心调用：

```python
temp_qwen.train_ppo(training_data, model_index=model_index)
```

输出：

```text
final_qwen
```

这一步之后得到的是变异后最终训练过的临时 Qwen 个体。

## train_ppo() 内部具体做什么

真实 PPO 训练函数在 `src/llm/qwen_module.py` 的 `QwenModule.train_ppo()` 中。

它从 `training_data` 中读取：

```python
sequences = torch.stack([d["sequences"] for d in batch_data])
action_mask = torch.stack([d["action_mask"] for d in batch_data])
r_scores = torch.stack([d["reward"] for d in batch_data])
```

然后做 token 对齐：

```python
action_mask_shifted = action_mask[:, 1:]

if r_scores.size(1) == sequences.size(1) and r_scores.size(1) > 1:
    r_scores_shifted = r_scores[:, 1:]
else:
    r_scores_shifted = r_scores
```

为什么要 shift？因为自回归语言模型预测第 `t+1` 个 token 时，logprob 和 loss 通常对应 `sequences[:, 1:]`。所以代码把 `action_mask` 和 reward 都对齐到被预测 token 的位置。

接着它计算旧策略、参考策略和 critic value：

```python
old_log_probs = self.actor(..., return_logprobs=True)
ref_log_probs = self.ref_model(..., return_logprobs=True)
values = self.critic(...)
```

KL 项为：

```python
kl = old_log_probs - ref_log_probs
```

最终进入 PPO 的 reward 是：

```python
combined_rewards = (-self.init_kl_coef * kl + r_scores_shifted) * action_mask_shifted.float()
```

也就是说，PPO 用的不是单纯扩散 reward，而是：

```text
扩散 reward - KL 惩罚
```

其中 `action_mask_shifted` 会把非生成部分的位置屏蔽掉。

然后计算优势和 returns：

```python
adv, ret = self.get_advantages_and_returns(values, combined_rewards)
```

最后分别更新 actor 和 critic：

```python
actor_loss = PolicyLoss(...)
critic_loss = ValueLoss(...)
actor_loss.backward()
critic_loss.backward()
self.actor_optim.step()
self.critic_optim.step()
```

所以 PPO 的真正输入是：

```text
sequences
action_mask
reward
```

PPO 的真正输出是：

```text
更新后的 actor 参数
更新后的 critic 参数
loss 数值
```

它不会直接输出 response。response 是训练前或训练后另外调用 `generate()` 得到的。

## response 是 step 级还是 token 级？

严格来说，本项目中 response 是“序列级文本输出”。

它不是 step 级，因为：

- `generate()` 返回的是完整字符串列表 `responses`。
- `step_4_1_generate_and_collect()` 没有把 response 拆成 `Step 1`、`Step 2` 的列表。
- `training_data` 中每条样本只有一个完整 `response` 字段。
- PPO loss 是根据 `sequences` 和 `action_mask` 在 token 维度计算的。

它也不是“每个 token 一个独立 response”。token 只是训练对齐和 loss 计算的粒度。

更准确的说法是：

```text
response 的样本粒度：整段序列
PPO 的优化粒度：token 位置
reward 的形状：序列级 reward 数组，通常和 token 序列长度对齐
```

## ppo_io_snapshot.json 示例说明

快照文件路径：

```text
outputs/ppo_io_snapshot/ppo_io_snapshot.json
```

它由 `export_ppo_io_snapshot.py` 生成，结构分为：

```json
{
  "first": {
    "stage": "first_ppo",
    "input": {},
    "output": []
  },
  "second": {
    "stage": "second_ppo",
    "input": {},
    "output": []
  }
}
```

### first：第一次 PPO 示例

`first.input` 中保存第一次 PPO 前的原始输入：

```json
{
  "prompts": [
    "What is 3 + 4?",
    "What is 9 - 5?"
  ],
  "labels": [
    "#### 7",
    "#### 4"
  ]
}
```

`first.output` 中每个元素表示一个样本。为了便于阅读，下面只展示第一条样本的部分字段：

```json
{
  "prompt": "What is 3 + 4?",
  "label": "#### 7",
  "response": "Step 1: solve the arithmetic.\nStep 2: verify the result.\n#### 7",
  "sequences": [86, 119, 104, 115, 35, 52, 61, 35, 118, 114, 111, 121],
  "action_mask": [true, true, true, true, true, true, true, true, true, true, true, true],
  "hidden_states": [
    [0.05975111946463585, 0.08253942430019379, -0.06164448335766792]
  ],
  "reward": [0.49807068705558777, 0.337282657623291, 0.40267717838287354]
}
```

字段解释：

- `prompt`：当前样本输入给 Qwen 的 prompt。
- `label`：参考答案。
- `response`：Qwen 生成的完整回答。
- `sequences`：response 对应的 token id 序列。
- `action_mask`：哪些 token 参与 PPO 动作 loss。
- `hidden_states`：该 response 对应的 Tau，扩散模型用它生成 reward。
- `reward`：扩散模型为该序列生成的 reward 数组。

### second：第二次 PPO 示例

`second.input` 比 `first.input` 多了两个关键字段：

```json
{
  "prompts": ["What is 3 + 4?", "What is 9 - 5?"],
  "labels": ["#### 7", "#### 4"],
  "mutated_rewards_seed": [[0.45, 0.51, 0.38]],
  "reused_training_seed": [
    {
      "prompt": "What is 3 + 4?",
      "label": "#### 7",
      "response": "Step 1: solve the arithmetic.\nStep 2: verify the result.\n#### 7",
      "sequences": [86, 119, 104, 115],
      "action_mask": [true, true, true, true],
      "reward": [0.45, 0.51, 0.38]
    }
  ]
}
```

其中：

- `mutated_rewards_seed`：step 5.3.1 从旧 elite 的 Tau 变异得到的 reward。
- `reused_training_seed`：step 5.3.2 用旧 response / sequences / action_mask 加上变异 reward 组成的训练数据。

`second.output` 表示 temp_qwen 经过变异 reward 训练后，重新生成 response、重新得到 Tau、再由扩散模型生成标准 reward 之后的数据：

```json
{
  "prompt": "What is 3 + 4?",
  "label": "#### 7",
  "response": "Step 1: solve the arithmetic.\nStep 2: verify the result.\n#### 7",
  "sequences": [86, 119, 104, 115, 35, 52, 61, 35],
  "action_mask": [true, true, true, true, true, true, true, true],
  "hidden_states": [[0.012, -0.044, 0.087]],
  "reward": [0.49, 0.28, 0.77]
}
```

这对应 step 5.3.4、5.3.5、5.3.6 之后喂给 step 5.3.7 的 `align_training_data`。

## 两次 PPO 输入输出对照表

| 阶段 | 输入 response 来源 | 输入 reward 来源 | 直接输入给 train_ppo 的字段 | PPO 输出 |
| --- | --- | --- | --- | --- |
| 第一次 PPO，step 4.4 | 当前 Qwen 对 prompt 的首次生成 | 扩散模型根据首次 `Tau` 生成的 `R` | `sequences`, `action_mask`, `reward` | 更新后的当前 Qwen |
| 第二次 PPO，step 5.3.3 | elite 旧 response | truncated diffusion 变异 reward | `sequences`, `action_mask`, `reward` | 初步更新后的 temp_qwen |
| 第二次 PPO，step 5.3.7 | temp_qwen 重新生成的新 response | 扩散模型根据新 `Tau` 生成的标准 reward | `sequences`, `action_mask`, `reward` | 最终更新后的 temp_qwen / mutant |

## 为什么 rho 不等于 PPO 的 reward

在流程里还有一个容易混淆的点：`rho` 不是 `train_ppo()` 的直接输出。

PPO 使用的是 reward 序列 `R`，它来自扩散模型。`rho` 通常是在后续评估阶段得到的整体分数，例如 ORM / metric 对重新生成 response 的评分，或者某些 no-evolution 配置里从 reward sequence 聚合得到的辅助分数。

因此：

```text
reward / R：PPO 训练用的 token 对齐奖励
rho：个体质量评估分数，用于选择、记录、排序
```

reward 正常不代表 rho 一定高；rho 为 0 也不代表 PPO 没有跑。它们属于不同阶段的信号。

## 如何重新生成快照

在 `Masdiff_change` 根目录运行：

```bash
.venv/bin/python export_ppo_io_snapshot.py
```

生成文件：

```text
outputs/ppo_io_snapshot/ppo_io_snapshot.json
```

如果希望接入真实 Qwen 而不是 mock，需要确保当前环境的 `transformers` / `openrlhf` 能识别本地 Qwen3 checkpoint，并且真实配置中的 `qwen_module.class_path` 指向 `src.llm.qwen_module.QwenModule`。

## 最后总结

本项目中 PPO 的核心是：用 Qwen 生成整段 response，提取该 response 的 token 序列和 hidden states；扩散模型根据 hidden states 生成 reward 序列；然后 PPO 在 token 维度上把 `sequences`、`action_mask` 和 `reward` 对齐，用 actor / critic loss 更新 Qwen。

所以要理解这个流程，最关键的是区分三个粒度：

```text
样本粒度：一条 prompt 对应一整段 response
训练粒度：response 内部的 token 位置
评估粒度：整个 response 或整个个体的 rho
```