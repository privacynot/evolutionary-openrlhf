# Beta概率建模（核心）

## 1. 为什么使用Beta分布

ρ 满足：

\[
\rho \in [0,1]
\]

因此不能用普通高斯回归。

我们建模为：

\[
\rho \sim Beta(\alpha, \beta)
\]

---

## 2. Beta分布概率密度

\[
p(\rho|\alpha,\beta)=\frac{\rho^{\alpha-1}(1-\rho)^{\beta-1}}{B(\alpha,\beta)}
\]

解释：

- α 控制靠近 1 的趋势
- β 控制靠近 0 的趋势
- B(α,β) 为归一化项

---

## 3. 模型输出

模型不直接预测ρ，而是输出：

- α（成功倾向）
- β（失败倾向）

---

## 4. Beta Head设计

\[
\alpha = softplus(W_\alpha z) + 1
\]

\[
\beta = softplus(W_\beta z) + 1
\]

其中：

- z = fusion(τ, R)

---

## 5. 损失函数

采用负对数似然：

\[
\mathcal{L} = -\log p(\rho|\alpha,\beta)
\]

含义：

👉 让真实ρ在预测分布中概率最大

---

## 6. 与MSE区别

| 方法 | 特点 |
|------|------|
| MSE | 只拟合点 |
| Beta | 拟合分布 |

Beta模型可以表达：

- 不确定性
- 多峰结构
- 极端样本鲁棒性
