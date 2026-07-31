"""
单张图片推理：输出植物类别（文件夹名）及概率。

用法:
  python predict.py --checkpoint runs/latest/best.pt --image path/to/one.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train import build_transforms, make_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt["meta"]
    class_names = meta["class_names"]
    backbone = meta["backbone"]
    img_size = meta.get("img_size", 224)

    num_classes = len(class_names)
    model = make_model(num_classes, backbone)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tf = build_transforms(img_size, train=False)
    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    topk = min(args.top_k, num_classes)
    vals, idx = torch.topk(prob, k=topk)

    print(f"图片: {args.image}")
    for i in range(topk):
        cname = class_names[idx[i].item()]
        print(f"  {i + 1}. {cname}  {vals[i].item():.4f}")


if __name__ == "__main__":
    main()
