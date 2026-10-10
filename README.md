# 植物显微图分类（ResNet50 全量微调）

基于 **ResNet50（ImageNet 预训练）全量微调** 的植物显微图像多分类项目，
包含 **K 折交叉验证训练** 与 **K 折模型集成（soft-voting）单图推理** 两个脚本。

- 训练：`trainv3_resnet50_full.py`
- 推理：`predictv3_resnet50.py`

---

## 1. 功能概览

| 脚本 | 作用 | 核心特性 |
| --- | --- | --- |
| `trainv3_resnet50_full.py` | 分层 K 折训练 | 确定性统一预处理、全量微调、分层学习率、余弦退火、Early Stopping、类别权重长尾修正、OOF 诊断报告 |
| `predictv3_resnet50.py` | 单张图片预测 | 自动读取 checkpoint 元信息、K 折 soft-voting 集成、Top-K 候选与置信度差输出 |

**关键设计**

- **全量微调**：ResNet50 全部参数从第 1 个 epoch 起参与训练，backbone 用小学习率、分类头用大学习率（分层学习率）。
- **统一确定性预处理**：训练与推理使用完全一致的预处理链路，避免训练/推理分布不一致。
- **类别不平衡处理**：使用 `1 / sqrt(类别频次)` 归一化到均值 1 的温和类别权重。
- **模型选择**：以 **Macro-F1** 为早停与保存指标（对长尾类别更公平）。
- **集成推理**：对每个 fold 的最佳模型输出 softmax 概率后取平均投票。

---

## 2. 环境要求

- **操作系统**：Windows / Linux（脚本中加入了 Windows 中文字体适配）
- **Python**：3.10（已验证 3.10.21）
- **GPU**：已验证 NVIDIA RTX 4090 + CUDA 12.6（CPU 也可运行，但训练很慢）

安装依赖（本项目验证环境为 conda 环境 `microscope`）：

```powershell
conda activate microscope
pip install -r requirements.txt
```

> ⚠️ `torchvision` 版本必须与 `torch` 严格匹配。
> 本项目验证组合为 `torch==2.7.1` + `torchvision==0.22.1`（CUDA 12.6 构建，即 `2.7.1+cu126`）。
> 如需按 CUDA 版本重装，请参考 `requirements.txt` 末尾的官方源安装命令。

> 📝 训练脚本会加载 Windows 中文字体 `C:\Windows\Fonts\msyh.ttc` 用于绘制混淆矩阵。
> 在非 Windows 环境下请修改脚本顶部的 `font_path`，否则绘图会报错。

---

## 3. 数据目录结构

训练脚本使用 `torchvision.datasets.ImageFolder`，要求**每个类别一个子文件夹**：

```text
显微镜像/
├── 艾叶/
│   ├── xxx.jpg
│   └── ...
├── 白芍/
│   └── ...
├── 甘草/
│   └── ...
└── ...（共 N 个类别子文件夹）
```

类别数与类别名称由脚本自动扫描得到，无需手动配置。

---

## 4. 训练

```powershell
python trainv3_resnet50_full.py `
    --data-dir "显微镜像" `
    --out-dir "runs/v3_resnet50_full"
```

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-dir` | 必填 | ImageFolder 根目录（每类一个子文件夹） |
| `--out-dir` | `runs/v3_resnet50_full` | 输出目录（checkpoint、指标、图表） |
| `--img-size` | `320` | 输入分辨率 |
| `--batch-size` | `16` | 批大小 |
| `--k-folds` | `5` | 分层交叉验证折数（每类样本数不得少于折数） |
| `--seed` | `42` | 随机种子 |
| `--num-workers` | `8` | DataLoader 进程数（Windows 异常时可降至 `4` 或 `0`） |
| `--disable-auto-crop` | 关闭 | 关闭预处理中的自动主体裁切（用于 A/B 对照） |
| `--epochs` | `50` | 每折最大训练轮数 |
| `--backbone-lr` | `3e-5` | backbone 学习率 |
| `--head-lr` | `1e-4` | 分类头学习率 |
| `--patience` | `10` | Early Stopping 容忍轮数 |
| `--min-delta` | `1e-4` | 判定指标改善的最小阈值 |

> 提示：若某类别图片数少于 `--k-folds`，脚本会直接报错退出，此时请减少折数。

### 训练输出文件

| 文件 | 说明 |
| --- | --- |
| `fold{i}_best.pt` | 第 i 折最佳模型权重 + `meta` 元信息 |
| `fold{i}_meta.json` | 第 i 折配置元信息（类别、预处理版本等） |
| `confusion_matrix_oof.csv` | OOF 混淆矩阵（原始计数） |
| `confusion_matrix_oof_normalized.png` | 行归一化混淆矩阵图 |
| `per_class_metrics.csv` | 每类 Precision / Recall / F1 / Support |
| `top_confusions.csv` | 最严重的 A → B 单向误判 Top-30 |
| `cv_summary.json` | 汇总指标（各折 Macro-F1、OOF 总体指标等） |

---

## 5. 推理（单张图片）

```powershell
python predictv3_resnet50.py `
    --checkpoint-dir "runs/v3_resnet50_full" `
    --image "显微镜像/甘草/xxxx.jpg" `
    --top-k 3
```

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--checkpoint-dir` | 必填 | 含 `fold1_best.pt ... foldN_best.pt` 的目录 |
| `--image` | 必填 | 待预测的单张显微图路径 |
| `--top-k` | `3` | 输出前 K 个候选类别 |
| `--k-folds` | `0` | `0` = 从 checkpoint 元信息自动推断折数；也可手动指定 |

### 输出示例

```text
============================================================
图片: 显微镜像/甘草/xxxx.jpg
模型: ResNet50 Full Fine-tuning, 5-fold ensemble
类别数: 68
设备: cuda
输入尺寸: 320x320
预处理: microscope_v2, auto_crop=True
------------------------------------------------------------
Top-3 预测结果:
  1. 甘草                0.9312  (93.12%)
  2. 黄芪                0.0310  (3.10%)
  3. 黄芩                0.0124  (1.24%)
------------------------------------------------------------
最终预测: 甘草
Top-1 概率: 0.9312 (93.12%)
Top1-Top2 概率差: 0.9002 (90.02 个百分点)
============================================================
```

> 集成前会校验各 fold 的 `backbone` / `class_names` / `img_size` / `auto_crop` / `preprocess_version` 是否一致，
> 不一致会直接报错，避免混用不同配置的模型产生错误结果。

---

## 6. 统一预处理流程（训练 / 推理一致）

两者共用同一套确定性预处理 `MicroscopePreprocess`：

1. **EXIF 方向校正 + 统一 RGB**；
2. **背景色估计**：取图像四周边框像素中位数作为背景色；
3. **自动主体裁切**（可选）：按与背景的 RGB 欧氏距离 + 暗像素补充生成前景掩码，
   取包围盒并外扩 `12%` 边距；检测不可靠（前景过少 / 区域过小）时**自动退回整图**；
4. **背景亮度轻度归一化**：整体增益对齐到目标亮度（增益限定在 `0.88 ~ 1.12`，避免过度修正）；
5. **保持纵横比 Pad 成正方形**：用估计背景色填充；
6. **Resize** 到 `img_size × img_size`（BICUBIC）。

训练阶段在其后追加**温和增强**：随机旋转 180°、水平/垂直翻转、
小幅平移缩放（`translate=0.05`、`scale=0.90~1.10`）、轻微 ColorJitter。
不使用 RandomCrop，避免裁掉关键显微结构。

---

## 7. 常见问题

- **`ModuleNotFoundError: No module named 'torchvision'`**
  未安装 torchvision，执行 `pip install -r requirements.txt`。
- **绘图报字体错误**
  非 Windows 系统请修改 `trainv3_resnet50_full.py` 顶部 `font_path` 指向本机中文字体。
- **DataLoader 多进程报错（Windows）**
  将 `--num-workers` 降为 `4` 或 `0`。
- **显存不足**
  降低 `--batch-size` 或 `--img-size`。
- **类别样本数少于折数**
  减少 `--k-folds`，或把过小的类别与相近类别合并。
