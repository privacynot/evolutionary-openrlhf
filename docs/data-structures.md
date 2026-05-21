# 核心数据结构

## Individual（个体）
每个 `Individual` 实例代表一个完整的语言模型策略及其关联数据。

**主要属性：**
- `id`：唯一标识符
- `model`：Qwen 模型（支持 LoRA）
- `tokenizer`：对应的分词器
- `trajectories`：list of dict，每项包含 `context` 和 `response`
- `tau`：张量，形状 `[N, seq_len, hidden_dim]`，记录生成时的隐藏状态
- `R`：张量，形状 `[N, seq_len]`，扩散模型生成的 token 级奖励
- `rho`：标量，当前策略在任务上的 ORM 分数

**关键方法：**
- `generate_with_hidden_states(dataset, cfg)` → `(trajectories, tau)`
- `set_training_data(trajectories, rewards)`
- `ppo_update(epochs=None)`
- `evaluate_orm(dataset, cfg)` → float
- `clone()` → Individual（深拷贝权重）

## Population（种群）
管理所有个体，提供选择、裁剪等操作。

**主要方法：**
- `add(ind)` / `add_all(inds)`：添加个体
- `select_elite(elite_ratio)` → list：按 ρ 降序返回前 `elite_ratio` 比例的个体
- `get_top_k_tau_R(k)` → `(list of tau, list of R)`：获取 k 个最优个体的隐状态和奖励，用于训练扩散模型
- `prune(max_size)`：保留 ρ 最高的 `max_size` 个个体，其余丢弃
- `best_rho()` → float：当前种群最佳分数

## DiffusionRewardModel（扩散奖励模型）
基于一维 UNet 的扩散模型，用于从隐状态预测奖励序列。

**初始化参数（来自 `diffusion_config.yaml`）：**
- `hidden_dim`：UNet 内部通道数
- `diffusion_steps`：扩散步数
- `condition_dim`：条件向量的维度（需与策略模型隐层大小一致）
- `learning_rate`：优化器学习率

**关键方法：**
- `conditionally_generate(tau)` → `R`：以 `tau` 为条件，通过反向扩散生成奖励序列。
- `truncated_diffusion(tau, R, steps, noise)` → `R'`：对 `R` 加噪后部分去噪，产生变异奖励。
- `train(taus, Rs)`：用给定的 `(tau, R)` 对训练扩散模型。

## EvolutionTrainer（进化训练器）
顶层调度器，负责执行阶段一（初始化）和阶段二（进化循环）。

**初始化时需传入所有配置字典，主要方法：**
- `form_initial_population()` → Population
- `evolution_loop(population)` → Population
- `run()`：完整流程入口

该训练器内部调用上述所有数据结构，并利用 Ray 实现分布式计算。
