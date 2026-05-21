# 开发扩展

欢迎贡献！以下是一些常见的扩展方向。

## 添加新的基座模型
修改 `configs/model_config.yaml` 中的 `base_policy`，并确保 `diffusion_config.yaml` 的 `condition_dim` 与模型的隐层维度匹配。

## 自定义变异算子
在 `diffusion_reward.py` 中增加新的变异方法（如基于梯度的变异），并集成到 `evolution_trainer.py` 的变异步骤中。

## 更复杂的 ORM 评估
在 `data_utils.py` 的 `compute_orm` 中添加数学表达式等价性判定、编译器反馈等高级评估逻辑。

## 并行化改进
完善 `ray_utils.py` 中的 `get_placement_group`，利用 Ray Placement Group 精细分配 GPU 资源，实现初始化种群的全并行执行。

## 调试模式
在启动脚本中增加 `--debug` 参数，输出中间隐状态、奖励序列等详细日志（目前为 TODO）。

## 贡献流程
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/foo`)
3. 提交更改 (`git commit -am 'Add some foo'`)
4. 推送到分支 (`git push origin feature/foo`)
5. 创建 Pull Request
