# 模块接口

以下简要介绍 `evolutionary_rlhf` 包中各模块的公开接口。

## `evolution_trainer.py` – 进化训练器
**类 `EvolutionTrainer`**  
- `__init__(configs)`：接收五个配置字典。
- `form_initial_population() -> Population`：生成初始种群。
- `evolution_loop(population) -> Population`：执行进化迭代。
- `run()`：一次性运行全部流程。

## `population.py` – 种群管理
**类 `Population`**  
- `add(individual)`, `add_all(individuals)`
- `select_elite(elite_ratio) -> list`
- `get_top_k_tau_R(k) -> (list, list)`
- `prune(max_size)`
- `best_rho() -> float`

## `individual.py` – 个体定义
**类 `Individual`**  
- `__init__(id, model_cfg, ppo_cfg)`
- `generate_with_hidden_states(dataset, task_cfg) -> (trajectories, tau)`
- `set_training_data(trajectories, rewards)`
- `ppo_update(epochs=None)`
- `evaluate_orm(dataset, task_cfg) -> float`
- `clone() -> Individual`

## `diffusion_reward.py` – 扩散奖励模型
**类 `DiffusionRewardModel`**  
- `__init__(config)`
- `conditionally_generate(tau) -> Tensor`
- `truncated_diffusion(tau, R, steps, noise) -> Tensor`
- `train(taus, Rs)`

## `custom_reward_fn.py` – 奖励函数适配
**函数 `create_custom_reward_fn(individual)`**  
返回一个符合 OpenRLHF 格式的 `reward_fn(samples, **kwargs)`，可被 PPO 训练器直接调用。

## `data_utils.py` – 数据工具
- `load_dataset(task_cfg)` → Dataset  
- `extract_hidden_states(model, tokenizer, prompts)` → Tensor  
- `compute_orm(model, tokenizer, dataset, task_cfg)` → float  

## `ray_utils.py` – Ray 工具
- `initialize_ray(num_gpus, num_nodes)`
- `get_placement_group(num_gpus_per_actor)`

> 所有接口的具体实现尚在完善中，当前代码为框架形态，标注了 `TODO` 的部分会逐步补全。
