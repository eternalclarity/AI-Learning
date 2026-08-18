# 第三周机器学习模型对比与树集成扩展实验报告

> 注意：表格中的数值必须来自你本地真实运行结果，不要提前填写固定答案。

## 1. 实验目的

本实验使用 scikit-learn 对 Wisconsin Breast Cancer Diagnostic 数据集完成二分类任务，在统一的数据划分和评价标准下比较 Logistic Regression、SVM、Random Forest 和 MLP 四种模型。

本实验主要目标：

1. 掌握监督学习二分类的基本流程；
2. 理解训练集、验证集和测试集的作用；
3. 掌握 Accuracy、Precision、Recall、F1 和 ROC-AUC；
4. 观察训练表现与验证表现的差异；
5. 学习模型选择、保存和最终测试流程。

---

## 2. 数据集介绍

数据集名称：Wisconsin Breast Cancer Diagnostic。

数据规模：

- 样本数：569；
- 特征数：30；
- 类别数：2。

本实验重新定义：

```text
0 = benign
1 = malignant
```

数据划分：

| 数据集 | 样本数 | 比例 | 作用 |
|---|---:|---:|---|
| 训练集 | 待填写 | 约 70% | 学习模型参数 |
| 验证集 | 待填写 | 约 15% | 比较并选择模型 |
| 测试集 | 待填写 | 约 15% | 最终独立评估 |

---

## 3. 数据预处理

对 Logistic Regression、SVM 和 MLP 使用 StandardScaler：

```text
z = (x - μ) / σ
```

标准化与模型通过 Pipeline 绑定，标准化器只在训练集上 `fit()`，从而避免验证集与测试集信息泄漏。

Random Forest 不进行标准化，因为树模型主要根据特征阈值进行划分，对特征线性缩放不敏感。

---

## 4. 模型设置

本实验分为核心四模型和新版课程扩展模型。核心四模型用于满足原学习计划，扩展模型用于对应新版课程 87～99 集。


### 4.1 Logistic Regression

- C：1.0；
- max_iter：5000；
- 输入预处理：StandardScaler。

### 4.2 SVM

- kernel：RBF；
- C：1.0；
- gamma：scale；
- probability：True；
- 输入预处理：StandardScaler。

### 4.3 Random Forest

- n_estimators：300；
- max_depth：None；
- random_state：42。

### 4.4 Decision Tree（新版扩展）

- criterion：entropy；
- max_depth：默认不限制，并通过深度扫描额外分析；
- 目的：观察单棵树的非线性能力与高方差。

### 4.5 Gradient Boosting（新版扩展）

- 实现：GradientBoostingClassifier；
- learning_rate：0.08；
- max_iter：200；
- 说明：用于实践 Boosting 思想，不等同于 XGBoost。

### 4.6 MLP

- hidden_layer_sizes：(64, 32)；
- activation：ReLU；
- solver：Adam；
- early_stopping：True；
- 输入预处理：StandardScaler。

---

## 5. 评价指标

本实验使用：

- Accuracy；
- Balanced Accuracy；
- Precision；
- Recall；
- Specificity；
- F1；
- ROC-AUC；
- Training Time。

主要模型选择指标：验证集 F1。

验证集 F1 相同时，依次比较 ROC-AUC 和 Recall。

---

## 6. 验证集实验结果

从以下文件复制真实结果：

```text
outputs/results/model_comparison.csv
```

| Model | Train F1 | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC | Training Time/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression |  |  |  |  |  |  |  |
| SVM |  |  |  |  |  |  |  |
| Random Forest |  |  |  |  |  |  |  |
| MLP |  |  |  |  |  |  |  |

最佳模型：`待填写`

选择原因：`待填写`

---

## 7. 结果分析

### 7.1 总体性能

请回答：

1. 哪个模型验证集 F1 最高？
2. 哪个模型验证集 Recall 最高？
3. 哪个模型 ROC-AUC 最高？
4. 排名是否一致？

分析：

`待填写`

### 7.2 训练与验证差距

请比较每个模型：

```text
train_f1 - val_f1
```

差距最大的模型：`待填写`

是否可能过拟合：`待填写`

### 7.3 混淆矩阵

图片：

```text
outputs/plots/best_model_validation_confusion_matrix.png
```

填写：

- TN：
- FP：
- FN：
- TP：

重点分析 FN 与 FP 的含义：

`待填写`

### 7.4 ROC 曲线

图片：

```text
outputs/plots/validation_roc_curves.png
```

分析不同模型排序能力：

`待填写`

---

## 8. 决策树与集成学习扩展分析

### 8.1 单棵树深度实验

结合 `decision_tree_depth_sweep.csv` 和 `decision_tree_depth_curve.png` 回答：

- 深度增加时训练 F1 如何变化？
- 验证 F1 在何处达到较好水平？
- 深树是否出现训练分数极高而验证下降？
- 这是否符合高方差特征？

### 8.2 单棵树与随机森林

填写并分析：

| 模型 | Train F1 | Validation F1 | 泛化差距 |
|---|---:|---:|---:|
| Decision Tree | 待填写 | 待填写 | 待填写 |
| Random Forest | 待填写 | 待填写 | 待填写 |

说明 Bootstrap、多棵树投票和随机特征为什么可能降低方差。

### 8.3 Random Forest 与 Gradient Boosting

比较两者的训练关系、验证表现、训练时间与可能风险。说明本实验的 Gradient Boosting 为什么不是 XGBoost。

### 8.4 特征重要性

引用 `random_forest_feature_importance.csv` 和对应图片。说明内置特征重要性不能解释为因果关系。

---

## 9. 偏差与方差分析

学习曲线命令示例：

```powershell
python bias_variance_experiment.py --model "Logistic Regression"
```

请填写：

- 最终训练 F1：
- 最终交叉验证 F1：
- 泛化差距：
- 是否可能高偏差：
- 是否可能高方差：

结合曲线分析：

`待填写`

---

## 10. 最终测试结果

运行：

```powershell
python evaluate_best_model.py
```

从以下文件填写：

```text
outputs/results/final_test_metrics.json
```

| Metric | Value |
|---|---:|
| Accuracy |  |
| Balanced Accuracy |  |
| Precision |  |
| Recall |  |
| Specificity |  |
| F1 |  |
| ROC-AUC |  |

测试集混淆矩阵：

```text
outputs/plots/final_test_confusion_matrix.png
```

分析：

`待填写`

---

## 11. 验证集与测试集对比

请回答：

1. 测试集 F1 是否接近验证集 F1？
2. 测试集 Recall 是否明显下降？
3. 验证集选出的模型是否在测试集保持良好表现？
4. 如果差异较大，可能原因是什么？

分析：

`待填写`

---

## 12. 实验局限性

建议至少包括：

1. 数据集规模较小；
2. 只使用一次固定划分；
3. 超参数没有充分搜索；
4. 没有外部独立数据；
5. 没有概率校准和阈值优化；
6. 教学数据结果不能直接代表真实部署表现。

补充：

`待填写`

---

## 13. 改进方向

可以考虑：

- 5 折交叉验证；
- GridSearchCV；
- PR 曲线和 Average Precision；
- 调整分类阈值；
- 分析特征重要性；
- 计算置信区间；
- 使用更具挑战的数据集。

本实验下一步最优先的改进：

`待填写`

---

## 14. 实验总结

请用自己的话总结：

1. 本周学会了什么；
2. 哪个概念最重要；
3. 哪个错误最容易犯；
4. 传统机器学习与第二周 PyTorch 训练有什么联系；
5. 下一周学习深度学习前，你还需要补什么。

总结：

`待填写`
