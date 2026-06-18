# 训练与进化流程

## 1. 初始化阶段

- 初始化M个Qwen模型（种群）
- 初始化Diffusion模型

---

## 2. 数据收集

每个模型生成：

\[
\tau_i, R_i
\]

并通过环境得到：

\[
\rho_i
\]

---

## 3. PPO训练

使用 R 更新策略：

\[
\pi_\theta \leftarrow PPO(\tau, R)
\]

---

## 4. Diffusion训练

学习：

\[
\tau \rightarrow R
\]

---

## 5. 进化选择

选择：

\[
Top-M(\rho)
\]

---

## 6. 变异机制

使用 diffusion 生成：

- R'
- 新探索策略

---

## 7. Predictor训练

学习：

\[
(\tau, R) \rightarrow \rho
\]
