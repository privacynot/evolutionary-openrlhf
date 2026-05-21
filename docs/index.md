# Evolutionary OpenRLHF

一个将 **进化算法** 与 **强化学习从人类反馈（RLHF）** 相结合的实验框架，基于 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 构建。

我们维护一个 LLM 策略种群，使用 **扩散模型** 作为可学习的奖励生成器。外层进化压力（ORM 分数）驱动探索，截断扩散产生稳定且多样的变异。

## 文档导航
- **[入门指南](project-overview.md)**：了解项目背景、整体架构，快速开始第一次训练。
- **[核心概念](algorithm.md)**：深入算法主流程与关键数据结构。
- **[模块接口](api-reference.md)**：查阅各 Python 模块的用途与主要 API。
- **[开发扩展](development.md)**：了解如何为框架贡献新的功能。
