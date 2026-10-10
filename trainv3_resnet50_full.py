from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


# Windows 中文字体
font_path = r"C:\Windows\Fonts\msyh.ttc"

font_manager.fontManager.addfont(font_path)
chinese_font = font_manager.FontProperties(fname=font_path)

plt.rcParams["font.family"] = chinese_font.get_name()
plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MicroscopePreprocess:
    """
    显微图统一预处理（确定性，不做随机增强）：

    1. EXIF方向校正 + RGB统一；
    2. 根据图像边缘估计背景颜色；
    3. 可选：自动寻找主要非背景组织区域，保留 margin 后裁切；
    4. 轻度统一背景亮度（只做整体增益，不做激进颜色归一化）；
    5. 保持纵横比，使用估计背景色 Pad 成正方形；
    6. Resize 到 img_size × img_size。

    目的：
    - 减少不同原图尺寸、主体位置、背景亮度造成的域差异；
    - 尽量让主要植物组织在画面中的占比更一致；
    - 避免 RandomCrop 裁掉关键结构。

    自动裁主体是保守策略：检测失败时自动退回整图，不会强制乱裁。
    """

    def __init__(
        self,
        img_size: int = 320,
        auto_crop: bool = True,
        crop_margin: float = 0.12,
        min_foreground_ratio: float = 0.01,
        background_target: float = 238.0,
    ):
        self.img_size = img_size
        self.auto_crop = auto_crop
        self.crop_margin = crop_margin
        self.min_foreground_ratio = min_foreground_ratio
        self.background_target = background_target

    @staticmethod
    def _estimate_background(arr: np.ndarray) -> np.ndarray:
        h, w, _ = arr.shape
        border = max(2, int(min(h, w) * 0.04))

        top = arr[:border, :, :].reshape(-1, 3)
        bottom = arr[-border:, :, :].reshape(-1, 3)
        left = arr[:, :border, :].reshape(-1, 3)
        right = arr[:, -border:, :].reshape(-1, 3)

        edges = np.concatenate([top, bottom, left, right], axis=0)
        return np.median(edges, axis=0)

    def _foreground_bbox(
        self,
        arr: np.ndarray,
        bg_rgb: np.ndarray,
    ):
        h, w, _ = arr.shape
        arr_f = arr.astype(np.float32)

        # 与背景颜色的 RGB 欧氏距离
        color_dist = np.linalg.norm(arr_f - bg_rgb.reshape(1, 1, 3), axis=2)

        # 自适应阈值：背景差异很小时至少使用 18，避免把噪点当成组织
        edge_noise = np.percentile(color_dist, 25)
        threshold = max(18.0, float(edge_noise) * 2.8)

        mask = color_dist > threshold

        # 对明显偏暗像素做补充，适应黄色/棕色/绿色组织
        gray = arr_f.mean(axis=2)
        bg_gray = float(bg_rgb.mean())
        mask |= gray < (bg_gray - 20.0)

        foreground_ratio = float(mask.mean())
        if foreground_ratio < self.min_foreground_ratio:
            return None

        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None

        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1

        bw = x1 - x0
        bh = y1 - y0

        # 如果检测区域过于极端，认为自动检测不可靠，退回整图
        if bw < 0.08 * w or bh < 0.08 * h:
            return None

        margin_x = int(bw * self.crop_margin)
        margin_y = int(bh * self.crop_margin)

        x0 = max(0, x0 - margin_x)
        y0 = max(0, y0 - margin_y)
        x1 = min(w, x1 + margin_x)
        y1 = min(h, y1 + margin_y)

        return x0, y0, x1, y1

    def _normalize_background_brightness(
        self,
        arr: np.ndarray,
        bg_rgb: np.ndarray,
    ) -> np.ndarray:
        """
        只做整体亮度增益，不做每通道白平衡。
        这样尽量保留显微图片原本的 RGB 相对颜色信息。
        """
        bg_luma = float(
            0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        )

        if bg_luma < 1.0:
            return arr

        gain = self.background_target / bg_luma
        # 防止对曝光差异做过度修正
        gain = float(np.clip(gain, 0.88, 1.12))

        out = np.clip(arr.astype(np.float32) * gain, 0, 255)
        return out.astype(np.uint8)

    @staticmethod
    def _pad_to_square(img: Image.Image, fill_rgb: tuple[int, int, int]):
        w, h = img.size
        side = max(w, h)

        canvas = Image.new("RGB", (side, side), fill_rgb)
        left = (side - w) // 2
        top = (side - h) // 2
        canvas.paste(img, (left, top))
        return canvas

    def __call__(self, img: Image.Image) -> Image.Image:
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.asarray(img)
        bg_rgb = self._estimate_background(arr)

        if self.auto_crop:
            bbox = self._foreground_bbox(arr, bg_rgb)
            if bbox is not None:
                img = img.crop(bbox)
                arr = np.asarray(img)
                bg_rgb = self._estimate_background(arr)

        arr = self._normalize_background_brightness(arr, bg_rgb)
        img = Image.fromarray(arr, mode="RGB")

        fill = tuple(int(np.clip(x, 0, 255)) for x in bg_rgb)
        img = self._pad_to_square(img, fill)

        img = img.resize(
            (self.img_size, self.img_size),
            resample=Image.Resampling.BICUBIC,
        )
        return img


def build_transforms(
    img_size: int = 320,
    train: bool = True,
    auto_crop: bool = True,
):
    """
    V2:
    - 统一显微预处理在最前面；
    - 取消 RandomCrop；
    - 训练阶段只做不会大面积破坏主体的增强。
    """
    preprocess = MicroscopePreprocess(
        img_size=img_size,
        auto_crop=auto_crop,
    )

    if train:
        return transforms.Compose(
            [
                preprocess,
                transforms.RandomRotation(
                    degrees=180,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(245, 245, 245),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.90, 1.10),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(245, 245, 245),
                ),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.05,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            preprocess,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def make_model(num_classes: int) -> nn.Module:
    """ImageNet 预训练 ResNet50，替换最终全连接层。"""
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def enable_full_finetuning(model: nn.Module) -> None:
    """全量微调：从第 1 个 epoch 开始训练 ResNet50 的全部参数。"""
    for p in model.parameters():
        p.requires_grad = True


def build_full_finetune_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
) -> torch.optim.Optimizer:
    """
    全量微调使用分层学习率：
    - backbone：较小学习率，避免快速破坏 ImageNet 预训练特征；
    - fc：较大学习率，快速适配新的显微图类别。
    注意：两组参数从训练开始都参与反向传播。
    """
    backbone_params = [
        p
        for name, p in model.named_parameters()
        if not name.startswith("fc.")
    ]
    head_params = list(model.fc.parameters())

    return torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": backbone_lr,
            },
            {
                "params": head_params,
                "lr": head_lr,
            },
        ],
        weight_decay=1e-4,
    )


def stratified_k_fold_val_indices(
    targets: list[int],
    k: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}

    for idx, t in enumerate(targets):
        by_class.setdefault(t, []).append(idx)

    val_folds: list[list[int]] = [[] for _ in range(k)]

    for c in sorted(by_class.keys()):
        idxs = by_class[c][:]
        rng.shuffle(idxs)
        n = len(idxs)

        if n < k:
            raise SystemExit(
                f"类别索引 {c} 仅有 {n} 张图，少于折数 {k}。"
                "请减少 --k-folds 或增加该类样本。"
            )

        sizes = [n // k] * k
        for i in range(n % k):
            sizes[i] += 1

        start = 0
        for fold in range(k):
            end = start + sizes[fold]
            val_folds[fold].extend(idxs[start:end])
            start = end

    return val_folds


def compute_class_weights(
    targets: list[int],
    train_indices: list[int],
    num_classes: int,
) -> torch.Tensor:
    """
    保留 V1 已验证的温和长尾修正：
        weight = 1 / sqrt(class_count)
    然后归一化到平均权重 = 1。
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)

    for idx in train_indices:
        counts[targets[idx]] += 1

    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise RuntimeError(f"训练折中缺失类别: {missing}")

    weights = 1.0 / torch.sqrt(counts)
    weights = weights / weights.mean()
    return weights


def confusion_matrix_from_lists(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm


def metrics_from_confusion_matrix(cm: torch.Tensor):
    cm_f = cm.float()
    tp = torch.diag(cm_f)

    support = cm_f.sum(dim=1)
    pred_count = cm_f.sum(dim=0)

    precision = tp / pred_count.clamp_min(1.0)
    recall = tp / support.clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    valid = support > 0

    macro_precision = precision[valid].mean().item() if valid.any() else 0.0
    macro_recall = recall[valid].mean().item() if valid.any() else 0.0
    macro_f1 = f1[valid].mean().item() if valid.any() else 0.0
    accuracy = tp.sum().item() / cm_f.sum().clamp_min(1.0).item()

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.long().tolist(),
    }


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    n = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)

    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
):
    model.eval()

    total_loss = 0.0
    n = 0

    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        pred = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    cm = confusion_matrix_from_lists(
        y_true,
        y_pred,
        num_classes,
    )

    metrics = metrics_from_confusion_matrix(cm)
    metrics["loss"] = total_loss / max(n, 1)

    return metrics, y_true, y_pred


def save_confusion_matrix_csv(
    cm: torch.Tensor,
    class_names: list[str],
    out_path: Path,
) -> None:
    with open(
        out_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)

        for i, name in enumerate(class_names):
            writer.writerow([name] + cm[i].tolist())


def save_confusion_matrix_png(
    cm: torch.Tensor,
    class_names: list[str],
    out_path: Path,
) -> None:
    """
    68 类时不在格子里写数字。
    绘制按真实类别逐行归一化的 OOF confusion matrix。
    """
    cm_plot = cm.float()
    row_sum = cm_plot.sum(dim=1, keepdim=True).clamp_min(1.0)
    cm_plot = cm_plot / row_sum

    fig_size = max(15, min(24, len(class_names) * 0.30))

    fig = plt.figure(
        figsize=(fig_size, fig_size)
    )
    ax = fig.add_subplot(111)

    im = ax.imshow(
        cm_plot.numpy(),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(
        "5-fold OOF Confusion Matrix (row-normalized)"
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))

    ax.set_xticklabels(
        class_names,
        rotation=90,
        fontsize=6,
    )
    ax.set_yticklabels(
        class_names,
        fontsize=6,
    )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=220,
    )
    plt.close(fig)


def save_per_class_metrics(
    metrics: dict,
    class_names: list[str],
    out_path: Path,
) -> None:
    with open(
        out_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "class",
                "support",
                "precision",
                "recall",
                "f1",
            ]
        )

        for i, name in enumerate(class_names):
            writer.writerow(
                [
                    name,
                    metrics["support"][i],
                    metrics["per_class_precision"][i],
                    metrics["per_class_recall"][i],
                    metrics["per_class_f1"][i],
                ]
            )


def save_top_confusions(
    cm: torch.Tensor,
    class_names: list[str],
    out_path: Path,
    top_n: int = 30,
) -> None:
    """
    直接输出最严重的 A -> B 单向误判。
    对 68 类任务比肉眼看完整矩阵更方便。
    """
    rows = []

    for i in range(len(class_names)):
        support = int(cm[i].sum().item())
        if support == 0:
            continue

        for j in range(len(class_names)):
            if i == j:
                continue

            count = int(cm[i, j].item())
            if count <= 0:
                continue

            rows.append(
                (
                    count,
                    count / support,
                    class_names[i],
                    class_names[j],
                    support,
                )
            )

    rows.sort(
        key=lambda x: (x[0], x[1]),
        reverse=True,
    )

    with open(
        out_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "true_class",
                "predicted_as",
                "error_count",
                "error_rate_within_true_class",
                "true_class_support",
            ]
        )

        for count, rate, true_name, pred_name, support in rows[:top_n]:
            writer.writerow(
                [
                    true_name,
                    pred_name,
                    count,
                    rate,
                    support,
                ]
            )


def main():
    p = argparse.ArgumentParser(
        description="植物显微图大类分类 V3 - ResNet50 全量微调"
    )

    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="ImageFolder 根目录：每种植物一个子文件夹",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/v3_resnet50_full"),
    )

    # 输入与 DataLoader
    p.add_argument(
        "--img-size",
        type=int,
        default=320,
        help="默认 320；RTX 4090 24GB 可稳定使用",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--k-folds",
        type=int,
        default=5,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help=(
            "默认 8，适配 i9-13900KF + Windows + RTX 4090；"
            "如遇 DataLoader/多进程异常可退回 4 或 0"
        ),
    )
    p.add_argument(
        "--disable-auto-crop",
        action="store_true",
        help="关闭统一预处理中的自动主体裁切，用于 A/B 对照",
    )

    # 全量微调
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    p.add_argument(
        "--backbone-lr",
        type=float,
        default=3e-5,
        help="ResNet50 backbone 学习率",
    )
    p.add_argument(
        "--head-lr",
        type=float,
        default=1e-4,
        help="最终 fc 分类头学习率",
    )

    # Early Stopping
    p.add_argument(
        "--patience",
        type=int,
        default=10,
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )

    args = p.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"device = {device}")

    auto_crop = not args.disable_auto_crop

    print(
        "统一预处理: "
        f"RGB + 背景估计 + "
        f"{'自动主体裁切 + ' if auto_crop else ''}"
        f"亮度轻度归一化 + "
        f"Pad正方形 + Resize {args.img_size}"
    )

    train_tf = build_transforms(
        img_size=args.img_size,
        train=True,
        auto_crop=auto_crop,
    )

    val_tf = build_transforms(
        img_size=args.img_size,
        train=False,
        auto_crop=auto_crop,
    )

    root = str(args.data_dir)

    ds_for_split = datasets.ImageFolder(
        root,
        transform=train_tf,
    )

    if len(ds_for_split.classes) < 2:
        raise SystemExit(
            "至少需要 2 个植物类别。"
        )

    num_classes = len(ds_for_split.classes)
    n_total = len(ds_for_split)

    print(f"类别数: {num_classes}")
    print(f"总图片数: {n_total}")

    targets = ds_for_split.targets

    class_counts = [0] * num_classes
    for t in targets:
        class_counts[t] += 1

    print("类别样本数:")
    for i, name in enumerate(ds_for_split.classes):
        print(
            f"  {name}: {class_counts[i]}"
        )

    k = args.k_folds

    if k < 2:
        raise SystemExit(
            "--k-folds 至少为 2"
        )

    val_folds = stratified_k_fold_val_indices(
        targets,
        k,
        args.seed,
    )

    all_idx_set = set(range(n_total))

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_best_macro_f1: list[float] = []

    oof_true: list[int] = []
    oof_pred: list[int] = []

    for fold_idx in range(k):
        val_idx = sorted(
            val_folds[fold_idx]
        )

        train_idx = sorted(
            all_idx_set
            - set(val_idx)
        )

        train_ds = Subset(
            datasets.ImageFolder(
                root,
                transform=train_tf,
            ),
            train_idx,
        )

        val_ds = Subset(
            datasets.ImageFolder(
                root,
                transform=val_tf,
            ),
            val_idx,
        )

        loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        }
        if args.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        model = make_model(
            num_classes
        ).to(device)

        class_weights = compute_class_weights(
            targets,
            train_idx,
            num_classes,
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device)
        )

        best_path = (
            args.out_dir
            / f"fold{fold_idx + 1}_best.pt"
        )

        meta = {
            "class_names": ds_for_split.classes,
            "class_to_idx": ds_for_split.class_to_idx,
            "backbone": "resnet50",
            "pretrained_weights": "IMAGENET1K_V2",
            "img_size": args.img_size,
            "fold": fold_idx + 1,
            "k_folds": k,
            "auto_crop": auto_crop,
            "preprocess_version": "microscope_v2",
            "selection_metric": "macro_f1",
            "class_weight_method": "inverse_sqrt_frequency_mean1",
            "full_finetuning": True,
            "epochs": args.epochs,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "num_workers": args.num_workers,
        }

        with open(
            args.out_dir
            / f"fold{fold_idx + 1}_meta.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                meta,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"\n==== Fold "
            f"{fold_idx + 1}/{k} "
            f"| train={len(train_ds)} "
            f"val={len(val_ds)} ===="
        )

        print(
            "class weights range: "
            f"{class_weights.min().item():.3f} "
            f"~ "
            f"{class_weights.max().item():.3f}"
        )

        # ==================================================
        # ResNet50 全量微调：全部参数从第 1 个 epoch 开始训练
        # ==================================================
        enable_full_finetuning(model)

        optimizer = build_full_finetune_optimizer(
            model,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
        )

        best_macro_f1 = -1.0
        best_epoch_label = ""
        no_improve = 0

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            val_metrics, _, _ = evaluate(
                model,
                val_loader,
                criterion,
                device,
                num_classes,
            )

            scheduler.step()
            current_f1 = val_metrics["macro_f1"]

            print(
                f"[Fold {fold_idx + 1}/{k}] "
                f"FullFT {epoch}/{args.epochs} | "
                f"train_loss={tr_loss:.4f} "
                f"train_acc={tr_acc:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"macro_P={val_metrics['macro_precision']:.4f} "
                f"macro_R={val_metrics['macro_recall']:.4f} "
                f"macro_F1={current_f1:.4f}"
            )

            if current_f1 > best_macro_f1 + args.min_delta:
                best_macro_f1 = current_f1
                best_epoch_label = f"fullft_epoch_{epoch}"
                no_improve = 0

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "meta": meta,
                        "val_metrics": val_metrics,
                        "best_metric": "macro_f1",
                        "best_macro_f1": best_macro_f1,
                        "epoch_label": best_epoch_label,
                    },
                    best_path,
                )

                print(
                    "  -> 保存最佳模型: "
                    f"Macro F1={best_macro_f1:.4f}"
                )
            else:
                no_improve += 1
                print(
                    "  -> Macro F1 未改善 "
                    f"({no_improve}/{args.patience})"
                )

                if no_improve >= args.patience:
                    print("  -> Early Stopping")
                    break

        # 每折最佳 checkpoint 做 OOF
        try:
            best_ckpt = torch.load(
                best_path,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            best_ckpt = torch.load(
                best_path,
                map_location=device,
            )

        model.load_state_dict(
            best_ckpt["model_state"]
        )

        final_metrics, fold_true, fold_pred = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
        )

        fold_best_macro_f1.append(
            final_metrics["macro_f1"]
        )

        oof_true.extend(fold_true)
        oof_pred.extend(fold_pred)

        print(
            f"Fold {fold_idx + 1} 最佳模型: "
            f"{best_ckpt.get('epoch_label', 'unknown')} | "
            f"Macro F1="
            f"{final_metrics['macro_f1']:.4f}"
        )

    # ======================================================
    # OOF Summary
    # ======================================================
    oof_cm = confusion_matrix_from_lists(
        oof_true,
        oof_pred,
        num_classes,
    )

    oof_metrics = metrics_from_confusion_matrix(
        oof_cm
    )

    save_confusion_matrix_csv(
        oof_cm,
        ds_for_split.classes,
        args.out_dir
        / "confusion_matrix_oof.csv",
    )

    save_confusion_matrix_png(
        oof_cm,
        ds_for_split.classes,
        args.out_dir
        / "confusion_matrix_oof_normalized.png",
    )

    save_per_class_metrics(
        oof_metrics,
        ds_for_split.classes,
        args.out_dir
        / "per_class_metrics.csv",
    )

    save_top_confusions(
        oof_cm,
        ds_for_split.classes,
        args.out_dir
        / "top_confusions.csv",
        top_n=30,
    )

    per_class_report = {}

    for i, name in enumerate(
        ds_for_split.classes
    ):
        per_class_report[name] = {
            "precision": oof_metrics[
                "per_class_precision"
            ][i],
            "recall": oof_metrics[
                "per_class_recall"
            ][i],
            "f1": oof_metrics[
                "per_class_f1"
            ][i],
            "support": oof_metrics[
                "support"
            ][i],
        }

    summary = {
        "num_classes": num_classes,
        "num_images": n_total,
        "k_folds": k,
        "backbone": "resnet50",
        "pretrained_weights": "IMAGENET1K_V2",
        "full_finetuning": True,
        "epochs": args.epochs,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "num_workers": args.num_workers,
        "img_size": args.img_size,
        "preprocess_version": "microscope_v2",
        "auto_crop": auto_crop,
        "selection_metric": "macro_f1",
        "fold_best_macro_f1": fold_best_macro_f1,
        "mean_fold_macro_f1": (
            sum(fold_best_macro_f1)
            / len(fold_best_macro_f1)
        ),
        "oof_accuracy": oof_metrics["accuracy"],
        "oof_macro_precision": oof_metrics[
            "macro_precision"
        ],
        "oof_macro_recall": oof_metrics[
            "macro_recall"
        ],
        "oof_macro_f1": oof_metrics[
            "macro_f1"
        ],
        "per_class": per_class_report,
    }

    with open(
        args.out_dir
        / "cv_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n========== "
        "5-fold ResNet50 全量微调完成 "
        "=========="
    )

    for i, score in enumerate(
        fold_best_macro_f1,
        start=1,
    ):
        print(
            f"Fold {i}: "
            f"best Macro F1 = {score:.4f}"
        )

    print(
        "Fold 平均 Macro F1 = "
        f"{summary['mean_fold_macro_f1']:.4f}"
    )

    print(
        "OOF Accuracy        = "
        f"{oof_metrics['accuracy']:.4f}"
    )

    print(
        "OOF Macro Precision = "
        f"{oof_metrics['macro_precision']:.4f}"
    )

    print(
        "OOF Macro Recall    = "
        f"{oof_metrics['macro_recall']:.4f}"
    )

    print(
        "OOF Macro F1        = "
        f"{oof_metrics['macro_f1']:.4f}"
    )

    print(
        "\n诊断文件:"
    )
    print(
        "  "
        + str(
            args.out_dir
            / "per_class_metrics.csv"
        )
    )
    print(
        "  "
        + str(
            args.out_dir
            / "top_confusions.csv"
        )
    )
    print(
        "  "
        + str(
            args.out_dir
            / "confusion_matrix_oof.csv"
        )
    )
    print(
        "  "
        + str(
            args.out_dir
            / "confusion_matrix_oof_normalized.png"
        )
    )


if __name__ == "__main__":
    main()
