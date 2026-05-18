# Evolutionary OpenRLHF

一个将**进化算法**与**强化学习**相结合的实验框架，基于 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 构建。我们维护一个 LLM 策略种群，使用**扩散模型**作为可学习的奖励生成器，通过外层进化压力（ORM 分数）驱动探索，同时利用截断扩散实现稳定、多样的变异。

## 核心思想
- **外层循环**：种群管理（选择、变异、对齐）
- **内层循环**：每个个体的 PPO 训练，使用扩散模型生成的奖励
- **扩散模型**：学习从隐状态到奖励序列的映射，由精英个体数据训练

## 快速导航
- [入门指南](getting-started.md)：安装、配置、运行一条龙
- [核心概念](core-concepts.md)：算法流程与伪代码
- [模块接口](api/evolution_trainer.md)：各模块使用方法
- [内置示例](examples.md)：命令行和 notebook 示例
- [开发扩展](development.md)：如何自定义组件

## 引用
如果此项目对您的研究有帮助，请引用我们的工作（待定）。
