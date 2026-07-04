"""
03_class_practice.py

本脚本用于练习 Python 类、对象、继承和方法。

在后续 PyTorch 学习中，你会经常看到类似这样的代码：

class MyModel(nn.Module):
    ...

所以提前熟悉 class 的写法非常重要。

本脚本模拟一个机器学习实验对象：
1. BaseExperiment 是基础实验类；
2. ClassificationExperiment 是分类实验类，继承 BaseExperiment；
3. 程序会创建一个实验对象，并模拟运行实验。

运行方式：

python week1_python_review/scripts/03_class_practice.py
"""


class BaseExperiment:
    """
    基础实验类。

    它保存所有实验都可能需要的基本信息：
    - 实验名称
    - 学习率
    - 训练轮数
    """

    def __init__(self, name: str, learning_rate: float, epochs: int):
        self.name = name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.history = []

    def show_config(self) -> None:
        """
        打印实验配置。
        """
        print("=" * 40)
        print("实验配置")
        print("=" * 40)
        print(f"实验名称: {self.name}")
        print(f"学习率: {self.learning_rate}")
        print(f"训练轮数: {self.epochs}")
        print("=" * 40)

    def run(self) -> None:
        """
        运行实验。

        这是一个基础方法，子类可以重写它。
        """
        print("BaseExperiment 的 run 方法被调用。")
        print("如果是具体实验，通常需要在子类中重写这个方法。")


class ClassificationExperiment(BaseExperiment):
    """
    分类实验类。

    继承自 BaseExperiment，并增加分类任务相关的信息：
    - 类别数量
    - 评估指标
    """

    def __init__(
        self,
        name: str,
        learning_rate: float,
        epochs: int,
        num_classes: int,
        metric: str,
    ):
        super().__init__(name, learning_rate, epochs)

        self.num_classes = num_classes
        self.metric = metric

    def show_config(self) -> None:
        """
        重写父类方法，打印更完整的分类实验配置。
        """
        super().show_config()
        print("分类任务配置")
        print("=" * 40)
        print(f"类别数量: {self.num_classes}")
        print(f"评估指标: {self.metric}")
        print("=" * 40)

    def run(self) -> None:
        """
        模拟运行分类实验。

        这里不是真的训练模型，只是模拟 loss 下降和 accuracy 上升。
        """
        print("开始运行分类实验...")

        loss = 1.0
        accuracy = 0.5

        for epoch in range(1, self.epochs + 1):
            loss = loss * 0.85
            accuracy = accuracy + 0.05

            if accuracy > 0.99:
                accuracy = 0.99

            record = {
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
            }

            self.history.append(record)

            print(
                f"Epoch [{epoch}/{self.epochs}] "
                f"loss={loss:.4f}, accuracy={accuracy:.4f}"
            )

        print("分类实验运行结束。")

    def get_best_result(self) -> dict:
        """
        获取 accuracy 最高的一轮结果。

        返回：
            最佳实验结果
        """
        if not self.history:
            return {}

        best_result = max(self.history, key=lambda x: x["accuracy"])
        return best_result


def main() -> None:
    experiment = ClassificationExperiment(
        name="week1_classification_demo",
        learning_rate=0.001,
        epochs=10,
        num_classes=2,
        metric="accuracy",
    )

    experiment.show_config()
    experiment.run()

    best_result = experiment.get_best_result()

    print("=" * 40)
    print("最佳结果")
    print("=" * 40)
    print(best_result)


if __name__ == "__main__":
    main()