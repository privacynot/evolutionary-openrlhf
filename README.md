# Evolutionary OpenRLHF

一个将 **进化算法** 与 **强化学习** 相结合的实验框架，基于 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 构建。

我们维护一个 LLM 策略种群，使用 **扩散模型** 作为可学习的奖励生成器。外层进化压力（ORM 分数）驱动探索，截断扩散产生稳定且多样的变异。

## 核心思想
- 外层循环：种群管理（选择、变异、对齐）
- 内层循环：个体 PPO 训练，使用扩散生成的奖励
- 扩散模型：从隐状态到奖励序列的映射，由精英数据训练

## 快速开始
1. 安装依赖
```bash
pip install -r requirements.txt
pip install -e .
pip install git+https://github.com/OpenRLHF/OpenRLHF.git
