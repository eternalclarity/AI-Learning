"""
一个自定义 Dataset 和 DataLoader。

1. 定义一个数据集类；
2. 规定数据集的长度；
3. 根据下标取出一个样本；
4. 使用 DataLoader 将数据分成多个批次。
"""


from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
# Dataset：    用来定义一个数据集应该如何存储和读取
# DataLoader： 用来对 Dataset 进行批量读取、打乱顺序等操作


class StudentDataset(Dataset):
    """定义 StudentDataset 类.一个用于二分类演示的学生数据集, 继承自 PyTorch 提供的 Dataset 类

    实现两个重要方法:
    1. __len__()：数据集中一共有多少个样本
    2. __getitem__()：根据下标如何取出一个样本

    每个学生有两个输入特征： [study_hours, attendance_rate]
    分别表示：
        study_hours： 学习时间。
        attendance_rate： 出勤率。

    标签 Labels：
        0 = 未通过考试
        1 = 通过考试
    """

    # 创建并初始化数据集中的特征和标签
    def __init__(self):
        # features: tensor[8,2] 创建所有学生的特征数据, 每一行代表一个学生, 每一列代表一个特征
        self.features = torch.tensor(
            [
                [1.5, 0.55],
                [2.0, 0.60],
                [2.5, 0.62],
                [3.0, 0.70],
                [4.0, 0.75],
                [5.0, 0.82],
                [6.0, 0.88],
                [7.0, 0.93],
            ],
            dtype=torch.float32,
        )

        # labels: tensor[8](一维） 创建每个学生对应的标签, features[0] 对应 labels[0]
        self.labels = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1],
            dtype=torch.long,   # torch.long 实际上就是 64 位整数 int64
        )

    # 当执行：len(dataset) 就会调用dataset.__len__()
    def __len__(self) -> int:
        """返回数据集中的样本数量"""
        return len(self.features)   # len(self.features) 返回第一维大小，也就是 8

    # 当执行：dataset[0] 就会调用dataset.__getitem__(0)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回一个学生的特征 Tensor 和对应标签."""
        return self.features[index], self.labels[index]     # tensor的切片返回的仍然是tensor类型


def create_dataloader(batch_size: int = 3, shuffle: bool = True) -> DataLoader:
    """
    根据自定义数据集创建一个 DataLoader.

        batch_size: int = 3 -> 每个批次默认包含 3 个样本
        shuffle: bool = True -> 默认在每一轮读取数据之前打乱样本顺序

        -> DataLoader 表示该函数返回一个 DataLoader 对象
    """
    dataset = StudentDataset()  # 创建 StudentDataset 对象
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # 创建并返回 DataLoader


def main():
    dataset = StudentDataset()  # 创建一个原始 StudentDataset 数据集对象
    dataloader = create_dataloader(batch_size=3, shuffle=True)  # 创建 DataLoader

    # 输出 dataset 的连个重写方法
    print(f"Dataset size: {len(dataset)}")
    print(f"First sample: {dataset[0]}")

    """
        遍历 DataLoader 中的所有批次
        
        dataloader 每次会返回：(一个批次的 features, 一个批次的 labels) -> (features, labels)
            - features: tensor[batch_size, 2]
            - labels: tensor[batch_size, 1]
            - batch_size 是 dataloader返回的
        
        enumerate(..., start=1)：  为每个批次添加编号，并从 1 开始编号 -> batch_index
    """
    for batch_index, features, labels in enumerate(dataloader, start=1):
        print("=" * 50)
        print(f"Batch {batch_index}")
        print("Features:")
        print(features)
        print(f"Feature shape: {tuple(features.shape)}")
        print("Labels:")
        print(labels)
        print(f"Label shape: {tuple(labels.shape)}")


if __name__ == "__main__":
    main()
