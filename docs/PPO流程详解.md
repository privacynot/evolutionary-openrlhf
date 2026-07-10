# Masdiff_change 中 PPO 流程详解

本文只分析 `Masdiff_change` 工程当前实际运行的 no-evolution 流程，也就是入口 `masdiff_change/runner_no_evolution.py`。这里需要先纠正一个容易混淆的点：在 `Masdiff_change` 的 no-evolution 主流程里，每个 `k` 轮、每个 `model_index` 个体只发生一次真正的 PPO 训练，也就是 `step_4_4_train_qwen_ppo()` 调用 `train_ppo()`。

代码里确实还有 `src/pipeline/steps_LLM.py` 中的 `step_5_3_3_train_temp_qwen()` 和 `step_5_3_7_train_temp_qwen()`，它们属于完整进化版流程里的 5.3 变异阶段；但 `Masdiff_change/runner_no_evolution.py` 没有调用这些函数。因此，不能把它们写成 `Masdiff_change` 当前真实训练中的“第二次 PPO”。

本文会严格按照当前 `Masdiff_change` 的真实路径说明：

1. PPO 在哪里发生。
2. PPO 之前的输入是什么。
3. PPO 训练时真正吃进去的数据是什么。
4. PPO 之后输出了什么。
5. 为什么 `step_4_5` 看起来像“第二次生成”，但不是第二次 PPO。
6. `outputs/ppo_io_snapshot/ppo_io_snapshot.json` 示例如何理解。
7. 如果以后要改造成 step 级 response，应该怎么改。

## 1. 当前 no-evolution 主流程

入口文件是：

```text
masdiff_change/runner_no_evolution.py
```

核心循环是：

```text
for k in range(1, K + 1):
    population = executor.map(lambda model_index: build_individual(k, model_index), list(range(M)))
    diffusion_model = step_5_1_train_diffusion_with_population(diffusion_model, population)
```

也就是说，每一轮 `k` 做两件事：

```text
1. 为每个 model_index 构建一个 individual。
2. 用这一轮 population 里的 Tau / R / rho 训练扩散模型。
```

真正的 Qwen PPO 训练发生在 `build_individual()` 里面。

`build_individual()` 的实际步骤是：

```text
step_3_init_qwen_model()
step_4_1_generate_and_collect()
step_4_2_generate_prm_reward()
step_4_3_build_ppo_data()
step_4_4_train_qwen_ppo()
step_4_5_generate_and_calc_orm()
step_4_build_individual()
```

其中只有 `step_4_4_train_qwen_ppo()` 是 PPO。

后面的 `step_4_5_generate_and_calc_orm()` 会再次调用 Qwen 生成 response，然后计算 rho；但它只是评估，不是 PPO 训练。

## 2. PPO 发生在哪里

PPO 调用链如下：

```text
masdiff_change/runner_no_evolution.py
    build_individual()
        step_4_4_train_qwen_ppo()
            current_qwen.train_ppo(training_data, model_index=model_index)
```

对应函数在：

```text
src/pipeline/steps_LLM.py
```

函数定义是：

```python
def step_4_4_train_qwen_ppo(
    qwen_module: Any,
    current_qwen: Any,
    training_data: Any,
    model_index: int | None = None,
) -> Any:
    current_qwen.train_ppo(training_data, model_index=model_index)
    return current_qwen
```

所以从流程角度看，PPO 的输入是 `training_data`，输出是被更新后的 `current_qwen`。

更准确地说：

```text
PPO 输入：training_data
PPO 输出：模型参数被更新后的 QwenModule
```

它不会直接输出新的 response。新的 response 需要在 PPO 之后再次调用 `generate()` 才能得到。

## 3. PPO 输入是怎样构造出来的

PPO 输入不是直接从数据集来的，而是经过 4.1、4.2、4.3 三步构造出来的。

### 3.1 原始数据输入

最开始的数据来自数据集，例如：

```text
data/openclaw_dialog_masdiff_balanced.jsonl
```

加载之后会被规范成：

```text
prompts
labels
```

其中：

- `prompts` 是给 Qwen 的输入问题或任务上下文。
- `labels` 是参考答案或参考反馈，主要用于后续评估 rho。

### 3.2 step_4_1：生成 response 并收集 PPO 原始字段

函数：

```text
step_4_1_generate_and_collect()
```

输入：

```text
qwen_module / current_qwen
prompts
labels
num_samples=N
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

`generate()` 返回：

```text
responses
hidden_states
sequences
action_mask
logits 可选
```

这一步会组装出 `ppo_data_list`，每条样本类似：

```json
{
  "prompt": "...",
  "label": "...",
  "response": "...",
  "sequences": [86, 119, 104, 115],
  "action_mask": [true, true, true, true],
  "logits": null
}
```

这里的 `response` 是 Qwen 一次生成的完整文本。它不是 step 列表，也不是 token 列表。token 列表是 `sequences`。

这一步输出：

```text
ppo_data
tau
logits
```

其中：

- `ppo_data`：后续 PPO 数据的基础。
- `tau`：也就是 `hidden_states`，会作为扩散模型生成 reward 的条件。
- `logits`：可选记录项，不是 PPO 必需输入。

### 3.3 step_4_2：扩散模型生成奖励 R

函数：

```text
step_4_2_generate_prm_reward()
```

输入：

```text
diffusion_model
tau / hidden_states
ppo_data
step_tag_ids 可选
```

核心逻辑：

```python
raw_rewards_seq = diffusion.generate_reward(hidden_states)
rewards = _normalize_tensor(raw_rewards_seq)
```

输出：

```text
rewards
```

通常情况下，`rewards` 是和 token 序列长度对齐的二维张量：

```text
[batch_size, sequence_length]
```

比如一个 batch 里有 2 个样本，每个样本长度 24，那么 reward 形状就是：

```text
[2, 24]
```

这说明扩散模型不是给整个 response 一个单独分数，而是生成一串 reward。

### 3.4 step_4_3：把 reward 塞回 PPO 数据

函数：

```text
step_4_3_build_ppo_data()
```

核心逻辑：

```python
for i, item in enumerate(ppo_data):
    item["reward"] = rewards[i]
```

这一步之后，真正传给 PPO 的 `training_data` 变成：

```json
{
  "prompt": "...",
  "label": "...",
  "response": "...",
  "sequences": [86, 119, 104, 115],
  "action_mask": [true, true, true, true],
  "logits": null,
  "reward": [0.49, 0.33, 0.40, 0.35]
}
```

因此 PPO 的直接输入字段是：

```text
sequences
action_mask
reward
```

`prompt`、`label`、`response` 也保存在样本里，但在真实 `train_ppo()` 里，核心训练计算直接读取的是 `sequences`、`action_mask`、`reward`。

## 4. train_ppo() 内部怎么用这些输入

真实实现位于：

```text
src/llm/qwen_module.py
```

函数：

```text
QwenModule.train_ppo()
```

它从 `training_data` 中取出：

```python
sequences = torch.stack([d['sequences'] for d in batch_data]).squeeze(1).to(self.device)
action_mask = torch.stack([d['action_mask'] for d in batch_data]).squeeze(1).to(self.device)
r_scores = torch.stack([d['reward'] for d in batch_data]).squeeze(1).to(self.device)
```

然后做自回归 token 对齐：

```python
action_mask_shifted = action_mask[:, 1:]

if r_scores.size(1) == sequences.size(1) and r_scores.size(1) > 1:
    r_scores_shifted = r_scores[:, 1:]
else:
    r_scores_shifted = r_scores
```

这个 shift 的意义是：语言模型通常用前面的 token 预测下一个 token，所以 loss 对齐的是 `sequences[:, 1:]` 这一侧。

之后它计算：

```python
old_log_probs = self.actor(...)
ref_log_probs = self.ref_model(...)
values = self.critic(...)
```

再计算 KL：

```python
kl = old_log_probs - ref_log_probs
```

真正进入优势估计的 reward 是：

```python
combined_rewards = (-self.init_kl_coef * kl + r_scores_shifted) * action_mask_shifted.float()
```

这表示 PPO 用的是：

```text
扩散模型 reward - KL 惩罚
```

而且只在 `action_mask` 标记的生成 token 上生效。

然后计算优势和 return：

```python
adv, ret = self.get_advantages_and_returns(values, combined_rewards)
```

最后更新 actor 和 critic：

```python
new_log_probs = self.actor(...)
actor_loss = self.actor_loss_fn(...)
new_values = self.critic(...)
critic_loss = self.critic_loss_fn(...)
actor_loss.backward()
critic_loss.backward()
self.actor_optim.step()
self.critic_optim.step()
```

所以从模型训练角度看：

```text
PPO 输入：sequences, action_mask, reward
PPO 中间量：old_log_probs, ref_log_probs, kl, values, combined_rewards, adv, ret
PPO 输出：更新后的 actor / critic 参数，以及一个 loss 数值
```

## 5. PPO 后的输出是什么

在 `Masdiff_change` 中，PPO 后紧接着执行：

```text
step_4_5_generate_and_calc_orm()
```

这一步输入的是训练后的 Qwen：

```text
trained_qwen
eval_data
metric
model_index
```

它会再次调用：

```python
qwen_module.generate(batch_prompts, return_hidden_states=True, model_index=model_index)
```

然后得到新的：

```text
responses
```

接着调用：

```python
metric.compute_rho(q={"labels": labels}, simulation_data=responses)
```

输出：

```text
generation_result
answer_rho
```

这一步非常重要，但它不是 PPO。

它的作用是：

```text
用 PPO 更新后的 Qwen 再生成一遍 response，然后让 metric / ORM 打分，得到 rho。
```

因此当前 `Masdiff_change` no-evolution 流程里可以区分为：

| 阶段 | 是否 PPO | 输入 | 输出 |
| --- | --- | --- | --- |
| 4.1 generate and collect | 否 | prompts, labels, current_qwen | response, sequences, action_mask, tau |
| 4.2 diffusion reward | 否 | tau | reward / R |
| 4.3 build PPO data | 否 | ppo_data, reward | training_data |
| 4.4 train PPO | 是 | training_data | 更新后的 Qwen |
| 4.5 ORM evaluation | 否 | 更新后的 Qwen, eval prompts, labels | 新 response, rho |
| 5.1 train diffusion | 否 | population 中的 tau, rewards, rho | 更新后的 diffusion model |

所以如果你说“两个 PPO 的输入输出”，在 `Masdiff_change` 当前执行路径里应该改成：

```text
一次 PPO 的输入输出 + PPO 后重新生成评估的输入输出
```

真实代码没有第二次 PPO。

## 6. ppo_io_snapshot.json 应该怎么看

快照文件是：

```text
outputs/ppo_io_snapshot/ppo_io_snapshot.json
```

它是用 `export_ppo_io_snapshot.py` 导出的，用来帮助观察 PPO 所需字段。

当前 JSON 顶层有：

```json
{
  "first": { ... },
  "second": { ... }
}
```

这里需要特别说明：这个快照里的 `second` 是为了展示“如果完整进化版 5.3 二次训练存在时，输入输出会长什么样”。它不是 `Masdiff_change/runner_no_evolution.py` 当前真实执行出来的第二次 PPO。

如果只分析 `Masdiff_change` no-evolution 主流程，真正对应的是 `first` 部分。

### first.input

示例：

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

这对应数据集加载后传入 4.1 的原始输入。

### first.output

示例中第一条样本大致是：

```json
{
  "prompt": "What is 3 + 4?",
  "label": "#### 7",
  "response": "Step 1: solve the arithmetic.\nStep 2: verify the result.\n#### 7",
  "sequences": [86, 119, 104, 115, 35, 52, 61, 35],
  "action_mask": [true, true, true, true, true, true, true, true],
  "hidden_states": [
    [0.05975111946463585, 0.08253942430019379, -0.06164448335766792]
  ],
  "reward": [0.49807068705558777, 0.337282657623291, 0.40267717838287354]
}
```

字段含义：

- `prompt`：输入给 Qwen 的问题。
- `label`：参考答案。
- `response`：Qwen 生成的完整回答。
- `sequences`：token id 序列，是 PPO 真正训练时读取的核心字段之一。
- `action_mask`：哪些 token 被视作生成动作。
- `hidden_states`：也就是 Tau，扩散模型根据它生成 reward。
- `reward`：扩散模型生成的 reward 序列，是 PPO 真正训练时读取的核心字段之一。

对应关系是：

```text
first.input.prompts / labels
    ↓ step_4_1_generate_and_collect
first.output.response / sequences / action_mask / hidden_states
    ↓ step_4_2_generate_prm_reward
first.output.reward
    ↓ step_4_3_build_ppo_data
training_data
    ↓ step_4_4_train_qwen_ppo
更新后的 Qwen
```

## 7. 当前 response 到底是什么粒度

当前 response 是“整段序列级 response”。

原因是：

- `qwen_module.generate()` 返回的是 `responses: list[str]`。
- `step_4_1_generate_and_collect()` 每个样本只保存一个 `response` 字段。
- 代码没有把 `response` 按 `Step 1`、`Step 2` 拆成列表。
- PPO 训练时读取的是 token 序列 `sequences`，不是 step 列表。

因此当前粒度应该这样描述：

```text
样本粒度：一条 prompt 对应一整段 response
训练对齐粒度：token
reward 形状：通常是 [batch_size, sequence_length]
rho 粒度：整条 response 或整个 individual 的评分
```

这也解释了为什么“response 看起来有 Step 1 / Step 2”并不意味着代码在做 step 级 PPO。它只是文本内容里包含步骤词，训练结构仍然是 token-level alignment。

## 8. 如果要改造成 step 级 response，应该怎么改

如果希望流程真正支持 step 级 response，不能只在 prompt 里要求模型写 `Step 1`、`Step 2`。必须在数据结构和训练链路里显式引入 step。

比较合理的改造方向如下。

### 8.1 数据集改成 step 列表

当前样本通常是：

```json
{
  "prompt": "...",
  "label": "..."
}
```

可以改成：

```json
{
  "prompt": "...",
  "steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "step_rewards": [0.3, 0.7, 1.0],
  "label": "..."
}
```

这样 response 不再只是一个字符串，而是有结构的 step list。

### 8.2 generate 阶段保存 step 边界

`step_4_1_generate_and_collect()` 需要增加 step 解析逻辑，例如按：

```text
\nStep 1:
\nStep 2:
\n<step>
```

或特殊 token 切分。

每个样本除了保存 `response`，还要保存：

```json
{
  "steps": ["...", "..."],
  "step_spans": [
    {"start": 0, "end": 12},
    {"start": 13, "end": 25}
  ]
}
```

其中 `step_spans` 表示每个 step 对应的 token 起止位置。

### 8.3 扩散 reward 改成 step 级或 step-to-token 映射

有两种选择。

第一种是直接让扩散模型输出 step 级 reward：

```text
R_step shape = [batch_size, num_steps]
```

然后在 PPO 前把每个 step 的 reward 广播到该 step 覆盖的 token：

```text
token_reward[t] = step_reward[i]，其中 t 属于 step_i 的 token span
```

第二种是扩散模型仍输出 token 级 reward，但在 loss 前按 step 聚合或平滑：

```text
step_reward[i] = mean(token_reward[start_i:end_i])
```

如果目标是真正解释 step 质量，第一种更清楚。

### 8.4 PPO training_data 增加 step 字段

可以把训练样本改成：

```json
{
  "prompt": "...",
  "label": "...",
  "response": "完整 response",
  "steps": ["Step 1: ...", "Step 2: ..."],
  "step_spans": [{"start": 0, "end": 10}, {"start": 11, "end": 22}],
  "sequences": [...],
  "action_mask": [...],
  "step_rewards": [0.2, 0.8],
  "reward": [...]
}
```

这里建议保留 `reward` 作为 token 级字段，因为当前 `train_ppo()` 已经按 token 对齐。新增 `step_rewards` 和 `step_spans` 用于解释、可视化和构造 token reward。

### 8.5 train_ppo 不一定要大改

如果最终仍把 step reward 展开成 token reward，那么 `train_ppo()` 可以基本不动：

```text
step_rewards + step_spans
    ↓ broadcast
reward[token]
    ↓
原 train_ppo()
```

这种改造风险最小。

如果要让 PPO loss 本身按 step 做 advantage，则改动会更大，需要把 logprob、value、reward 都按 step span 聚合：

```text
step_logprob = sum(token_logprob[start:end])
step_value = mean(value[start:end])
step_reward = reward_for_this_step
```

然后用 step 序列做 GAE 和 PPO loss。这会更符合“step 级 PPO”，但实现复杂度和调试难度都更高。

## 9. 最准确的一句话总结

在当前项目里，PPO 只有一次：

```text
step_4_4_train_qwen_ppo(training_data)
```

它的输入是由 `response -> sequences/action_mask` 和 `Tau -> diffusion reward` 合并出的 `training_data`；它的输出是更新后的 Qwen 模型参数。PPO 后的 `step_4_5` 会再次生成 response 并计算 rho。

如果未来要做真正 step 级 response，需要显式保存 `steps`、`step_spans`、`step_rewards`，再把 step reward 映射到 token reward 或重写 PPO 为 step-level advantage/loss。
