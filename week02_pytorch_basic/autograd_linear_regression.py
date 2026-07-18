"""
一个简单的线性回归实验练习 PyTorch 自动求导 autograd。
- pytorch中的 grad就是损失函数对模型参数的 偏微分
- 整个pytorch 反向传播的过程就是多元函数微分链式求导法则求偏微分的过程
- loss.backward() 就是求偏微分的过程

数据满足：
    y = 2x + 3

训练方式：
1. 直接使用 Tensor 的梯度更新，自己创建 w,b，手动更新参数。
2. 使用 nn.Linear 模型和 torch.optim.SGD 优化器更新参数。
3. 程序对两者的训练结果进行对比，并将训练结果图保存
4. 结果相同，因为手动就是模拟了nn.Linear的函数式 y = wx + b 与 MSE Loss函数 和 optim.SGD 的 参数更新方法 x -= llr * grad.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from utils import set_seed

BASE_DIR = Path(__file__).resolve().parent


def create_dataset(sample_count: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """创建带有噪声的 y = 2x + 3 线性数据。"""

    x = torch.linspace(-1.0, 1.0, sample_count).reshape(-1, 1)  # 在 -1.0 到 1.0 之间生成 sample_count 个均匀分布的数字。
    noise = torch.randn_like(x) * 0.15  # 和 x 形状相同的随机 Tensor
    y = 2.0 * x + 3.0 + noise  # 再加上 noise，使数据不完全落在一条直线上
    return x, y


def train_manual(
        x: torch.Tensor,
        y: torch.Tensor,
        epochs: int,
        learning_rate: float,
) -> tuple[float, float, list[float]]:
    """
    通过手动读取梯度并更新参数来训练标量 w 和 b， 没用优化器 optimizer
        - 过程
        1. 初始化 w, b
        2. 每轮: 计算预测值 pre
        3. 计算损失函数 loss = (pre - y) ** 2
        4. 计算参数的梯度 w.grad, b.grad
        5. 由梯度，学习率更新参数值
        6. 清理本轮的参数梯度

        - 函数返回三个结果：
         1. 训练得到的权重 w；
         2. 训练得到的偏置 b；
         3. 每一轮的损失列表。
    """

    w = torch.randn(1, requires_grad=True)  # 创建一个包含一个随机数的 Tensor
    b = torch.zeros(1, requires_grad=True)  # 偏置 b 初始值设置为 0
    loss_history: list[float] = []  # 保存每轮训练的损失

    print("\nManual autograd training")
    for epoch in range(1, epochs + 1):
        predictions = w * x + b  # 根据当前参数计算预测值, PyTorch 会通过广播机制完成运算
        loss = torch.mean((predictions - y) ** 2)  # 计算均方误差 MSE, 对所有平方误差求平均值

        loss.backward()  # 反向传播 loss对 w 的梯度 w.grad.item() 和 loss 对 b 的梯度 b.grad.item()

        # 暂时关闭自动求导, 更新参数 -> 梯度下降
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # PyTorch 默认会累加梯度.因此完成一次参数更新后，需要手动清空 w 的梯度
        w.grad.zero_()
        b.grad.zero_()

        loss_history.append(loss.item())  # loss.item() 将它转换成普通 Python 浮点数

        # 控制日志输出频率,第一轮一定输出,每完成总训练轮数的 10% 输出一次
        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            print(
                f"Epoch {epoch:4d}/{epochs} | "  # :4d 表示整数至少占 4 个字符宽度。
                f"loss={loss.item():.6f} | "
                f"w={w.item():.4f} | b={b.item():.4f}"  # :.4f 表示保留 4 位小数。
            )

    return w.detach().item(), b.detach().item(), loss_history


def train_with_optimizer(
        x: torch.Tensor,
        y: torch.Tensor,
        epochs: int,
        learning_rate: float,
) -> tuple[nn.Linear, list[float]]:
    """
    定义使用 PyTorch 模型和优化器进行训练的函数, 使用 MSELoss 和 SGD 训练 nn.Linear
        - 声明
            1. x: tensor[-1, 1]
            2. y: tensor[-1, 1]
            3. model: nn.Linear
            4. loss_fn: nn.MSELoss
            5. optimizer: torch.optim.SGD

        - 过程
            1. 前向传播， 计算 pre
            2. 计算损失函数 loss , nn.MSEloss()
            3. 清空上一轮参数的梯度 optimizer.zero_grad()
            4. 反向传播，计算参数梯度
            5. 更新参数， torch.optim.SGD

        - 返回：
             1. 训练后的 nn.Linear 模型；
             2. 每一轮的损失列表。
    """
    # 创建一个全连接线性层
    # in_features=1：每个样本有一个输入特征 x,  out_features=1：每个样本输出一个预测值 y。
    model = nn.Linear(in_features=1, out_features=1)

    loss_fn = nn.MSELoss()  # 创建 均方误差(Mean Squared Error) 损失函数

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   # 创建 随机 梯度下降 (Stochastic Gradient Descent) 优化器 -> 梯度下降,从而在 loss-w,b 的三维曲面函数中下山 -> loss下降

    loss_history: list[float] = []

    print("\nOptimizer training")
    # 开始训练
    for epoch in range(1, epochs + 1):
        predictions = model(x)  # 1. 前向传播

        loss = loss_fn(predictions, y)  # 2. 使用均方误差计算预测值和真实值之间的损失

        optimizer.zero_grad()   # 3. 清空上一轮保存的梯度

        loss.backward()     # 4. 反向传播，自动计算模型参数的梯度 -> loss.item()获取loss值用于累加

        optimizer.step()    # 5. 根据刚才计算出的梯度更新模型参数

        loss_history.append(loss.item())

        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            learned_w = model.weight.item()
            learned_b = model.bias.item()
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"loss={loss.item():.6f} | "
                f"w={learned_w:.4f} | b={learned_b:.4f}"
            )

    return model, loss_history


def save_regression_figure(
        x: torch.Tensor,
        y: torch.Tensor,
        manual_parameters: tuple[float, float],  # 手动训练得到的参数，包含 w 和 b
        optimizer_model: nn.Linear, # 使用优化器训练得到的模型。
        save_path: Path,
) -> None:
    """定义保存线性回归结果图的函数"""

    # parents=True：如果上层目录不存在，也一起创建,   exist_ok=True：如果目录已经存在，不报错。
    save_path.parent.mkdir(parents=True, exist_ok=True)

    manual_w, manual_b = manual_parameters  # 对元组进行解包, 解包后分别得到手动训练的权重和偏置

    # 绘图预测阶段不需要计算梯度，所以使用 torch.no_grad() 关闭梯度跟踪
    with torch.no_grad():
        manual_prediction = manual_w * x + manual_b     # 使用手动训练得到的 w 和 b 计算预测值。
        optimizer_prediction = optimizer_model(x)       # 使用 nn.Linear 模型计算预测值。

    plt.figure(figsize=(8, 5))   # 创建一个宽 8 英寸、高 5 英寸的画布

    plt.scatter(x.numpy(), y.numpy(), alpha=0.7, label="Samples")    # 绘制原始数据散点图
    plt.plot(x.numpy(), manual_prediction.numpy(), label="Manual autograd") # 绘制手动 autograd 方法拟合出的直线
    # plt.plot(x.numpy(), optimizer_prediction.numpy(), label="Optimizer")    # 绘制优化器方法拟合出的直线

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear regression with PyTorch autograd")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()   # 自动调整图表布局，避免文字被裁剪
    plt.savefig(save_path, dpi=200)  # dpi=200 表示图片分辨率为每英寸 200 个点
    plt.close() # 关闭当前画布，释放内存


def parse_args() -> argparse.Namespace:
    """
    定义命令行参数解析函数。
    """
    parser = argparse.ArgumentParser(description="PyTorch autograd linear regression")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=BASE_DIR / "outputs" / "plots" / "linear_regression_fit.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 设置随机种子, torch.randn()等随机操作每次运行时尽量产生相同结果
    set_seed(args.seed)

    #  创建训练数据
    x, y = create_dataset()

    # 使用手动梯度更新方法进行训练
    manual_w, manual_b, _ = train_manual(
        x=x,
        y=y,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    # 使用 nn.Linear 和 SGD 优化器训练
    optimizer_model, _ = train_with_optimizer(
        x=x,
        y=y,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    # 训练参数结果对比
    print("\nFinal comparison")
    print(f"Target relationship:      w=2.0000, b=3.0000")
    print(f"Manual autograd result:   w={manual_w:.4f}, b={manual_b:.4f}")
    print(
        "Optimizer result:         "
        f"w={optimizer_model.weight.item():.4f}, "
        f"b={optimizer_model.bias.item():.4f}"
    )

    # 训练结果画图
    save_regression_figure(
        x=x,
        y=y,
        manual_parameters=(manual_w, manual_b),
        optimizer_model=optimizer_model,
        save_path=args.save_path,
    )
    print(f"Figure saved to: {args.save_path}")


if __name__ == "__main__":
    main()
