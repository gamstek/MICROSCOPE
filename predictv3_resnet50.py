from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


# ============================================================
# 与 trainv3_resnet50_full.py 完全一致的确定性验证/推理预处理
# ============================================================
class MicroscopePreprocess:
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

    def _foreground_bbox(self, arr: np.ndarray, bg_rgb: np.ndarray):
        h, w, _ = arr.shape
        arr_f = arr.astype(np.float32)

        color_dist = np.linalg.norm(
            arr_f - bg_rgb.reshape(1, 1, 3),
            axis=2,
        )

        edge_noise = np.percentile(color_dist, 25)
        threshold = max(18.0, float(edge_noise) * 2.8)
        mask = color_dist > threshold

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
        bg_luma = float(
            0.299 * bg_rgb[0]
            + 0.587 * bg_rgb[1]
            + 0.114 * bg_rgb[2]
        )

        if bg_luma < 1.0:
            return arr

        gain = self.background_target / bg_luma
        gain = float(np.clip(gain, 0.88, 1.12))

        out = np.clip(arr.astype(np.float32) * gain, 0, 255)
        return out.astype(np.uint8)

    @staticmethod
    def _pad_to_square(
        img: Image.Image,
        fill_rgb: tuple[int, int, int],
    ) -> Image.Image:
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


def build_inference_transform(
    img_size: int,
    auto_crop: bool,
):
    """严格对应训练脚本的 train=False 验证变换。"""
    return transforms.Compose(
        [
            MicroscopePreprocess(
                img_size=img_size,
                auto_crop=auto_crop,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def make_resnet50(num_classes: int) -> nn.Module:
    """
    推理时无需重新下载/加载 ImageNet 权重；
    checkpoint 会覆盖完整网络参数。
    """
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(
        description=(
            "植物显微图 V3 - ResNet50 全量微调 "
            "K-fold ensemble 单图预测"
        )
    )

    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="包含 fold1_best.pt ... foldN_best.pt 的目录",
    )
    p.add_argument(
        "--image",
        type=Path,
        required=True,
        help="待预测的单张显微图",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="输出前 K 个候选类别；68 类任务默认 3",
    )
    p.add_argument(
        "--k-folds",
        type=int,
        default=0,
        help=(
            "0=自动从 fold1 checkpoint 的 meta 读取；"
            "也可手动指定，例如 --k-folds 5"
        ),
    )

    args = p.parse_args()

    if not args.image.exists():
        raise SystemExit(f"图片不存在: {args.image}")

    first_path = args.checkpoint_dir / "fold1_best.pt"
    if not first_path.exists():
        raise SystemExit(f"缺少 checkpoint: {first_path}")

    first_ckpt = load_checkpoint(first_path)
    if "meta" not in first_ckpt:
        raise SystemExit("fold1_best.pt 中缺少 meta，无法安全预测。")

    first_meta = first_ckpt["meta"]

    backbone = first_meta.get("backbone")
    if backbone != "resnet50":
        raise SystemExit(
            f"checkpoint backbone={backbone!r}，"
            "本预测脚本要求 ResNet50。"
        )

    class_names = first_meta.get("class_names")
    if not class_names:
        raise SystemExit("checkpoint meta 中缺少 class_names。")

    num_classes = len(class_names)
    img_size = int(first_meta.get("img_size", 320))
    auto_crop = bool(first_meta.get("auto_crop", True))
    preprocess_version = first_meta.get(
        "preprocess_version",
        "unknown",
    )
    full_finetuning = first_meta.get("full_finetuning", None)

    if args.k_folds > 0:
        k_folds = args.k_folds
    else:
        k_folds = int(first_meta.get("k_folds", 5))

    checkpoint_paths = [
        args.checkpoint_dir / f"fold{i}_best.pt"
        for i in range(1, k_folds + 1)
    ]

    missing = [str(p) for p in checkpoint_paths if not p.exists()]
    if missing:
        raise SystemExit(
            "缺少 checkpoint:\n  " + "\n  ".join(missing)
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tf = build_inference_transform(
        img_size=img_size,
        auto_crop=auto_crop,
    )

    with Image.open(args.image) as img:
        x = tf(img)

    x = x.unsqueeze(0).to(device, non_blocking=True)

    probs_all: list[torch.Tensor] = []

    for ckpt_path in checkpoint_paths:
        ckpt = load_checkpoint(ckpt_path)
        meta = ckpt.get("meta", {})

        # ----------------------------------------------------
        # ensemble 安全检查：5 个 fold 必须完全同构
        # ----------------------------------------------------
        if meta.get("backbone") != "resnet50":
            raise SystemExit(
                f"{ckpt_path.name}: backbone 不是 resnet50。"
            )

        if meta.get("class_names") != class_names:
            raise SystemExit(
                f"{ckpt_path.name}: 类别名称/顺序不一致，"
                "不能做 ensemble。"
            )

        if int(meta.get("img_size", img_size)) != img_size:
            raise SystemExit(
                f"{ckpt_path.name}: img_size 不一致。"
            )

        if bool(meta.get("auto_crop", auto_crop)) != auto_crop:
            raise SystemExit(
                f"{ckpt_path.name}: auto_crop 配置不一致。"
            )

        if meta.get("preprocess_version", preprocess_version) != preprocess_version:
            raise SystemExit(
                f"{ckpt_path.name}: preprocess_version 不一致。"
            )

        model = make_resnet50(num_classes)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(device)
        model.eval()

        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        probs_all.append(prob.cpu())

        # 逐 fold 释放显存，避免长期批量预测时累积
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 5-fold soft voting：平均每个模型的类别概率
    ensemble_prob = torch.stack(
        probs_all,
        dim=0,
    ).mean(dim=0)

    topk = max(1, min(args.top_k, num_classes))
    vals, idx = torch.topk(ensemble_prob, k=topk)

    top1_prob = float(vals[0].item())
    top1_name = class_names[idx[0].item()]

    if num_classes >= 2:
        best2_vals, _ = torch.topk(ensemble_prob, k=2)
        top2_prob = float(best2_vals[1].item())
        margin = top1_prob - top2_prob
    else:
        top2_prob = 0.0
        margin = top1_prob

    print("=" * 60)
    print(f"图片: {args.image}")
    print(
        "模型: ResNet50 Full Fine-tuning, "
        f"{len(checkpoint_paths)}-fold ensemble"
    )
    print(f"类别数: {num_classes}")
    print(f"设备: {device}")
    print(f"输入尺寸: {img_size}x{img_size}")
    print(
        f"预处理: {preprocess_version}, "
        f"auto_crop={auto_crop}"
    )
    if full_finetuning is not None:
        print(f"full_finetuning: {full_finetuning}")

    print("-" * 60)
    print(f"Top-{topk} 预测结果:")

    for rank in range(topk):
        cname = class_names[idx[rank].item()]
        prob = float(vals[rank].item())
        print(
            f"  {rank + 1}. {cname:<20} "
            f"{prob:.4f}  ({prob * 100:.2f}%)"
        )

    print("-" * 60)
    print(f"最终预测: {top1_name}")
    print(f"Top-1 概率: {top1_prob:.4f} ({top1_prob * 100:.2f}%)")
    print(
        "Top1-Top2 概率差: "
        f"{margin:.4f} ({margin * 100:.2f} 个百分点)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
