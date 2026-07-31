"""
显微镜像植物分类 — 训练脚本

数据目录（ImageFolder 格式）:
  DATA_ROOT/
    植物A/   # 文件夹名即类别标签，可用中文或英文
      *.png
      *.jpg
    植物B/
      ...

用法:
  python train.py --data-dir ./data --epochs 80 --batch-size 16 --k-folds 5

训练仅支持分层 K 折交叉验证：每一折的训练/验证集中各类比例与全数据一致。
每类至少需 k 张图；增加新植物时在 data 下新建文件夹并放入图片后重新训练。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.15, 0.15, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def make_model(num_classes: int, backbone: str = "resnet50") -> nn.Module:
    if backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"未知 backbone: {backbone}")
    return m


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def stratified_k_fold_val_indices(targets: list[int], k: int, seed: int) -> list[list[int]]:
    """按类别分层：每类样本尽量均匀分到 k 份，返回每折的验证集索引（各类在全数据中的比例在各折中保持一致）。"""
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
                f"类别索引 {c} 仅有 {n} 张图，少于折数 {k}，无法进行分层 K 折。"
                "请减少 --k-folds 或增加该类样本。"
            )
        sizes = [n // k] * k
        for i in range(n % k):
            sizes[i] += 1
        start = 0
        for f in range(k):
            end = start + sizes[f]
            val_folds[f].extend(idxs[start:end])
            start = end
    return val_folds


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    """返回平均 loss、每类准确率列表、宏平均（各类准确率算术平均，无样本的类不计入）。"""
    model.eval()
    total_loss = 0.0
    n = 0
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    per_class_total = torch.zeros(num_classes, dtype=torch.long)
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        n += x.size(0)
        for c in range(num_classes):
            mask = y == c
            per_class_total[c] += mask.sum().item()
            per_class_correct[c] += ((pred == y) & mask).sum().item()
    avg_loss = total_loss / max(n, 1)
    per_class_acc: list[float] = []
    macro_vals: list[float] = []
    for c in range(num_classes):
        t = per_class_total[c].item()
        if t > 0:
            a = per_class_correct[c].item() / t
            per_class_acc.append(a)
            macro_vals.append(a)
        else:
            per_class_acc.append(float("nan"))
    macro_acc = sum(macro_vals) / len(macro_vals) if macro_vals else 0.0
    return avg_loss, per_class_acc, macro_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True, help="ImageFolder 根目录（每类一个子文件夹）")
    p.add_argument("--out-dir", type=Path, default=Path("runs/latest"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    p.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="分层 K 折交叉验证折数（每折训练一个模型；各类在 train/val 中比例与全数据一致）",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0, help="Windows 下建议 0")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = str(args.data_dir)
    train_tf = build_transforms(args.img_size, train=True)
    val_tf = build_transforms(args.img_size, train=False)

    ds_for_split = datasets.ImageFolder(root, transform=train_tf)
    if len(ds_for_split.classes) < 2:
        raise SystemExit("至少需要 2 个类别文件夹才能训练。")

    n_total = len(ds_for_split)
    k = args.k_folds
    if k < 2:
        raise SystemExit("--k-folds 至少为 2；本脚本仅支持 K 折交叉验证。")
    if n_total < k:
        raise SystemExit(f"样本数 {n_total} 小于折数 {k}，无法进行 K 折交叉验证。")

    targets = ds_for_split.targets
    val_folds = stratified_k_fold_val_indices(targets, k, args.seed)
    all_idx_set = set(range(n_total))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_best_macro: list[float] = []

    for fold_idx in range(k):
        val_idx = val_folds[fold_idx]
        train_idx = sorted(all_idx_set - set(val_idx))

        train_ds = Subset(datasets.ImageFolder(root, transform=train_tf), train_idx)
        val_ds = Subset(datasets.ImageFolder(root, transform=val_tf), val_idx)
        full_ds = ds_for_split

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        num_classes = len(full_ds.classes)
        model = make_model(num_classes, args.backbone).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        fold_best_macro_acc = -1.0
        best_path = args.out_dir / f"fold{fold_idx + 1}_best.pt"

        meta = {
            "class_names": full_ds.classes,
            "class_to_idx": full_ds.class_to_idx,
            "backbone": args.backbone,
            "img_size": args.img_size,
            "fold": fold_idx + 1,
            "k_folds": k,
        }
        with open(args.out_dir / f"fold{fold_idx + 1}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"==== 第 {fold_idx + 1}/{k} 折（分层），训练 {len(train_ds)} / 验证 {len(val_ds)} ====")
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, per_cls_acc, macro_acc = evaluate(
                model, val_loader, criterion, device, num_classes
            )
            scheduler.step()
            cls_parts = [
                f"{full_ds.classes[c]}={per_cls_acc[c]:.4f}"
                for c in range(num_classes)
                if per_cls_acc[c] == per_cls_acc[c]
            ]
            cls_str = "  ".join(cls_parts)
            print(
                f"[Fold {fold_idx + 1}/{k}] Epoch {epoch}/{args.epochs}  "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
                f"val_loss={va_loss:.4f}  验证各类准确率: {cls_str}"
            )
            if macro_acc > fold_best_macro_acc:
                fold_best_macro_acc = macro_acc
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "meta": meta,
                        "val_macro_acc": macro_acc,
                        "val_per_class_acc": {
                            full_ds.classes[c]: per_cls_acc[c]
                            for c in range(num_classes)
                            if per_cls_acc[c] == per_cls_acc[c]
                        },
                        "epoch": epoch,
                    },
                    best_path,
                )
                print(
                    f"  -> 折 {fold_idx + 1} 保存最佳（宏平均 val_macro_acc={macro_acc:.4f}）"
                )

        fold_best_macro.append(fold_best_macro_acc)

    mean_macro = sum(fold_best_macro) / len(fold_best_macro)
    print("分层 K 折完成。每折最佳宏平均验证准确率（各类准确率算术平均）：")
    for i, acc in enumerate(fold_best_macro, start=1):
        print(f"  Fold {i}: {acc:.4f}")
    print(f"折间平均: {mean_macro:.4f}")


if __name__ == "__main__":
    main()
