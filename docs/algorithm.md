# 算法主流程

Evolutionary OpenRLHF 的算法可概括为以下两个阶段：

## 阶段一：初始化种群
1. 从基座模型（如 Qwen2.5-7B）克隆出 M 个独立个体。
   population = []
   base_model = load_base_model("Qwen2.5-7B")
   diffusion_model = RandomInitDiffusion()  # 随机初始化的扩散模型
2. 对每个个体 i：
   1. 使用该策略在 N 个 prompt 上生成回答，并记录每步的隐藏状态 `Tau`。
   2. 用一个随机初始化的扩散模型，以 `Tau` 为条件生成奖励序列 `R`。
   3. 将 `(context, response, R)` 组装成 PPO 训练数据，执行一次 PPO 更新。
   4. 在标准答案上评估更新后的策略，得到 ORM 分数 `ρ`。
   model = clone_model(base_model)          # 从基座模型克隆出一个独立个体
   tau, responses = [], []
   for prompt in prompt_dataset[:N]:
       resp, hidden = model.generate(prompt, return_hidden_states=True)
       responses.append(resp)
       tau.append(hidden)                   # hidden 是每一步的隐藏状态列表
   tau = stack_hidden_states(tau)          # 堆叠成统一张量，形状 [N, seq_len, hidden_dim]
   R = diffusion_model.sample_conditioned_on(tau)   # 形状与 Tau 的时间步维度匹配
3. 得到初始种群，每个个体携带其权重、`Tau`、`R` 和 `ρ`。
   individual = {
    "model": model,
    "tau": tau,
    "R": R,
    "rho": rho
   }
   population.append(individual)

## 阶段二：进化迭代（重复 K 代）
每一代执行以下步骤：

### 1. 训练扩散模型
选取当前种群中 ρ 最高的 `top_k` 个个体，用它们的 `(Tau, R)` 对训练扩散模型（监督学习，让扩散模型学会从隐状态预测高分奖励）。
   sorted_pop = sorted(population, key=lambda x: x["rho"], reverse=True)
   top_individuals = sorted_pop[:top_k]
### 2. 选择精英种群
根据 ρ 对种群排序，保留前 `elite_ratio` 比例的个体进入精英池。
   sorted_population = sorted(population, key=lambda x: x["rho"], reverse=True)
   num_elites = int(len(population) * elite_ratio)
   elites = sorted_population[:num_elites]
### 3. 对每个精英个体进行变异与对齐
对精英池中的每个个体，生成一个变异后代：

- **变异**：对精英的奖励序列 `R` 应用截断扩散——先加噪，再反向去噪若干步，得到变异奖励 `R'`。
  for elite in elites:
    # 3a. 变异：截断扩散
    # 先加噪到中间步 t_mutate
    t_mutate = T // 2   # 示例：加噪到一半时间步
    R_noisy = diffusion_model.q_sample(elite["R"], t_mutate)
    # 反向去噪若干步（从 t_mutate 到 0 的不完全去噪，可提前停止）
    R_prime = diffusion_model.partial_denoise(R_noisy, t_mutate, stop_step=stop_early_step,
                                              condition=elite["tau"])
- **初步训练**：用 `R'` 对精英策略进行 PPO 训练，得到一个临时策略。
   # 3b. 初步训练：用变异奖励 R' 对精英策略进行 PPO 更新（得到临时策略）
    temp_model = clone_model(elite["model"])
    ppo_data_mut = build_ppo_batch(prompts=prompt_dataset,
                                   responses=collect_responses(elite["model"], prompt_dataset),
                                   rewards=R_prime)
    PPOTrainer(temp_model).step(ppo_data_mut)
- **探索新轨迹**：使用临时策略生成新的回答和隐状态 `τ_new`。
  tau_new, responses_new = [], []
    for prompt in prompt_dataset:
        resp, hidden = temp_model.generate(prompt, return_hidden_states=True)
        responses_new.append(resp)
        tau_new.append(hidden)
    tau_new = stack_hidden_states(tau_new)
- **对齐**：将 `τ_new` 送入扩散模型，获得干净的标准奖励 `R_new`，并以此再执行一次 PPO 更新，确保策略与当前扩散模型的评估一致。
  R_new = diffusion_model.sample_conditioned_on(tau_new)
- **评估**：最终生成答案并计算新的 ORM 分数 `ρ_new`。
   rho_new = evaluate_orm(aligned_model, eval_prompts, ground_truth_answers)
一个精英个体变异后产生一个新个体，包含新的权重、`τ_new`、`R_new` 和 `ρ_new`。
   new_offspring.append({
        "model": aligned_model,
        "tau": tau_new,
        "R": R_new,
        "rho": rho_new
    })

### 4. 种群更新
将所有新产生的个体加入种群，保留 ρ 最高的 M 个个体，形成新一代种群。
   combined_population = population + new_offspring
   combined_population.sort(key=lambda x: x["rho"], reverse=True)
   population = combined_population[:M]

### 5. 记录
记录当前种群的最佳 ρ，用于监控进化趋势。
   best_rho = population[0]["rho"]
   print(f"Generation {gen}: best rho = {best_rho:.4f}")
   wandb.log({"generation": gen, "best_rho": best_rho})
