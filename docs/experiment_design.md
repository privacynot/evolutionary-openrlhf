# 实验设计与指标

## 1. 预测任务

目标：

\[
p(\rho|\tau,R)
\]

---

## 2. 评价指标

### （1）预测误差

- MSE（baseline）

---

### （2）排序能力

- Spearman correlation
- Kendall tau

---

### （3）分布拟合能力

- NLL（Beta likelihood）

---

## 3. 消融实验

### (1) 去掉R

测试：

\[
(\tau) \rightarrow \rho
\]

---

### (2) 去掉τ

测试：

\[
(R) \rightarrow \rho
\]

---

### (3) fusion方式对比

- concat
- attention

---

## 4. 核心假设验证

验证：

> τ 比 logits 更重要
> R 是否提升预测能力
