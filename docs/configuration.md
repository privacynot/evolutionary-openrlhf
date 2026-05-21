# 配置系统

所有超参数均通过 YAML 文件管理，存放在 `configs/` 目录下。你可以直接编辑这些文件来调整实验设置，无需修改代码。主要配置文件有：

| 文件 | 作用 |
|------|------|
| `evolution_config.yaml` | 种群规模、进化代数、精英比例等进化超参数 |
| `ppo_config.yaml` | PPO 训练的学习率、batch size、KL 惩罚等 |
| `model_config.yaml` | 基座模型路径、LoRA 设置、序列长度等 |
| `diffusion_config.yaml` | 扩散模型的架构、训练步数、条件维度等 |
| `task_config.yaml` | 数据集名称、ORM 评分方式、评估样本数等 |

---

## 1. 进化配置 (`evolution_config.yaml`)

```yaml
evolution:
  population_size: 16          # M: 种群中的个体总数
  num_generations: 20          # K: 进化迭代次数
  elite_ratio: 0.25            # 每代选为精英的个体比例 (0~1)
  top_k_for_diffusion: 8       # 训练扩散模型时使用的最高分个体数量
  checkpoint_dir: "./checkpoints" # 模型与种群状态的保存路径
  seed: 42                     # 随机种子

mutation:
  truncation_steps: 50         # 截断扩散的步数，越大变异越强
  noise_level: 0.05            # 加噪强度，控制奖励序列的扰动幅度
  alignment_ppo_epochs: 1      # 对齐阶段 PPO 更新的轮数
