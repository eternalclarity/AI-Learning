"""训练一轮、评估一轮和收集预测结果的通用函数。"""

from __future__ import annotations

import time  # 用于统计每个 epoch 的耗时。
from dataclasses import dataclass  # 用结构化对象返回一轮结果。

import torch  # 用于张量计算、自动求导和混合精度。
from torch import nn  # 用于模型与损失函数类型标注。
from torch.utils.data import DataLoader  # 用于 DataLoader 类型标注。
from tqdm import tqdm  # 显示训练和评估进度条。


@dataclass(frozen=True)
class EpochMetrics:
    """保存一轮训练或评估的聚合指标。"""

    loss: float  # 全部样本的平均损失。
    accuracy: float  # 全部样本的分类准确率。
    correct: int  # 预测正确的样本数。
    samples: int  # 实际处理的样本总数。
    seconds: float  # 本轮耗时。


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> EpochMetrics:
    """训练模型一个 epoch。"""

    model.train()  # 启用 Dropout，并让 BatchNorm 使用当前批次统计量。
    start_time = time.perf_counter()  # 记录高精度开始时间。
    total_loss = 0.0  # 累加“批次平均损失 × 批次样本数”。
    total_correct = 0  # 累加预测正确样本数。
    total_samples = 0  # 累加处理过的样本数。
    amp_enabled = use_amp and device.type == "cuda"  # 只有 CUDA 环境才真正启用 float16 AMP。

    progress_bar = tqdm(
        data_loader,
        desc=f"Train [{epoch}/{total_epochs}]",
        leave=False,
    )

    for images, labels in progress_bar:  # 每次从 DataLoader 取出一个 mini-batch。
        images = images.to(device, non_blocking=True)  # 将图片移动到 CPU 或 GPU。
        labels = labels.to(device, non_blocking=True)  # 将标签移动到同一设备。
        batch_size = labels.size(0)  # 获取当前批次真实样本数量。

        optimizer.zero_grad(set_to_none=True)  # 清空旧梯度；set_to_none 通常更节省内存。

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)  # 第一步：前向传播。
            loss = criterion(logits, labels)  # 第二步：计算交叉熵损失。

        if scaler is not None and amp_enabled:  # AMP 开启时使用梯度缩放避免下溢。
            scaler.scale(loss).backward()  # 第三步：缩放损失后反向传播。
            scaler.step(optimizer)  # 第四步：在确认梯度有效后更新参数。
            scaler.update()  # 动态调整下一批次的缩放因子。
        else:
            loss.backward()  # 普通精度下直接反向传播。
            optimizer.step()  # 根据当前梯度更新模型参数。

        predictions = logits.argmax(dim=1)  # 取最大 logit 对应的类别作为预测。
        total_loss += loss.item() * batch_size  # 按样本数加权累加损失。
        total_correct += (predictions == labels).sum().item()  # 统计当前批次正确数。
        total_samples += batch_size  # 更新总样本数。

        running_loss = total_loss / total_samples  # 计算当前累计平均损失。
        running_accuracy = total_correct / total_samples  # 计算当前累计准确率。
        progress_bar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_accuracy:.4f}")

    elapsed = time.perf_counter() - start_time  # 计算本轮耗时。

    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
        correct=total_correct,
        samples=total_samples,
        seconds=elapsed,
    )


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    description: str = "Evaluate",
) -> EpochMetrics:
    """在验证集或测试集上评估模型。"""

    model.eval()  # 关闭 Dropout，并让 BatchNorm 使用训练阶段累计的运行统计量。
    start_time = time.perf_counter()  # 记录评估开始时间。
    total_loss = 0.0  # 累加总损失。
    total_correct = 0  # 累加预测正确样本数。
    total_samples = 0  # 累加处理过的样本数。

    progress_bar = tqdm(data_loader, desc=description, leave=False)  # 创建评估进度条。

    for images, labels in progress_bar:  # 逐批读取验证或测试数据。
        images = images.to(device, non_blocking=True)  # 将图片移动到模型设备。
        labels = labels.to(device, non_blocking=True)  # 将标签移动到模型设备。
        batch_size = labels.size(0)  # 当前批次大小。

        logits = model(images)  # 只执行前向传播，不构建梯度图。
        loss = criterion(logits, labels)  # 计算当前批次损失。
        predictions = logits.argmax(dim=1)  # 得到离散预测类别。

        total_loss += loss.item() * batch_size  # 按样本数累计损失。
        total_correct += (predictions == labels).sum().item()  # 累计正确数。
        total_samples += batch_size  # 累计总样本数。

        running_loss = total_loss / total_samples  # 当前累计平均损失。
        running_accuracy = total_correct / total_samples  # 当前累计准确率。
        progress_bar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_accuracy:.4f}")

    elapsed = time.perf_counter() - start_time  # 计算评估耗时。

    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
        correct=total_correct,
        samples=total_samples,
        seconds=elapsed,
    )


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_images_to_keep: int = 32,
) -> dict[str, torch.Tensor]:
    """收集测试集真实标签、预测标签、概率和少量样例图片。"""

    model.eval()  # 确保预测行为稳定。
    all_labels: list[torch.Tensor] = []  # 保存每批真实标签。
    all_predictions: list[torch.Tensor] = []  # 保存每批预测标签。
    all_probabilities: list[torch.Tensor] = []  # 保存每批 Softmax 概率。
    sample_images: list[torch.Tensor] = []  # 只保存少量图片，避免占用大量内存。
    kept_images = 0  # 记录已经保存的样例图片数量。

    for images, labels in tqdm(data_loader, desc="Collect predictions", leave=False):
        images_on_device = images.to(device, non_blocking=True)  # 将图片移动到模型设备。
        logits = model(images_on_device)  # 得到 logits。
        probabilities = torch.softmax(logits, dim=1)  # 转换为每类概率。
        predictions = probabilities.argmax(dim=1)  # 取得概率最大的类别。

        all_labels.append(labels.cpu())  # 标签保留在 CPU，便于后续分析。
        all_predictions.append(predictions.cpu())  # 预测结果移回 CPU。
        all_probabilities.append(probabilities.cpu())  # 概率移回 CPU。

        remaining = max_images_to_keep - kept_images  # 计算还需要保存多少张样例图。
        if remaining > 0:
            images_to_keep = images[:remaining].cpu()  # 只截取需要的前几张。
            sample_images.append(images_to_keep)  # 保存标准化后的图片。
            kept_images += images_to_keep.size(0)  # 更新已保存数量。

    result = {
        "labels": torch.cat(all_labels, dim=0),  # 拼接成 [N]。
        "predictions": torch.cat(all_predictions, dim=0),  # 拼接成 [N]。
        "probabilities": torch.cat(all_probabilities, dim=0),  # 拼接成 [N,10]。
        "sample_images": torch.cat(sample_images, dim=0) if sample_images else torch.empty(0),
    }

    return result  # 返回后续绘图和保存所需的全部预测数据。
