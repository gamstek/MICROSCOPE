# MICROSCOPE
广东农检显微镜像

# train.py
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

# predict.py
