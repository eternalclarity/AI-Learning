"""检查分层划分是否互不重叠、完整且保持类别比例。"""

from __future__ import annotations

import unittest  # 使用 Python 标准测试框架。

import numpy as np  # 用于构造标签和统计类别数量。

from data_utils import create_stratified_split_indices  # 导入待测试函数。


class StratifiedSplitTests(unittest.TestCase):
    """测试分层划分的核心性质。"""

    def test_split_is_complete_disjoint_and_balanced(self) -> None:
        targets = np.repeat(np.arange(10), 100)  # 构造十个类别、每类 100 个样本。
        train_indices, val_indices = create_stratified_split_indices(
            targets=targets,
            val_ratio=0.1,
            seed=42,
        )

        train_set = set(train_indices)  # 转换为集合便于检查交集。
        val_set = set(val_indices)  # 转换为集合便于检查交集。

        self.assertEqual(len(train_indices), 900)  # 90% 样本进入训练集。
        self.assertEqual(len(val_indices), 100)  # 10% 样本进入验证集。
        self.assertTrue(train_set.isdisjoint(val_set))  # 两个集合不能有重叠。
        self.assertEqual(train_set | val_set, set(range(1000)))  # 两个集合合并应覆盖全部样本。

        train_counts = np.bincount(targets[train_indices], minlength=10)  # 统计训练集每类数量。
        val_counts = np.bincount(targets[val_indices], minlength=10)  # 统计验证集每类数量。
        self.assertTrue(np.all(train_counts == 90))  # 每类 90 个训练样本。
        self.assertTrue(np.all(val_counts == 10))  # 每类 10 个验证样本。

    def test_same_seed_produces_same_indices(self) -> None:
        targets = np.repeat(np.arange(3), 20)  # 构造简单三分类标签。
        first = create_stratified_split_indices(targets, val_ratio=0.2, seed=7)  # 第一次划分。
        second = create_stratified_split_indices(targets, val_ratio=0.2, seed=7)  # 相同种子再次划分。
        self.assertEqual(first, second)  # 两次结果必须完全一致。


if __name__ == "__main__":
    unittest.main()  # 允许直接运行当前测试文件。
