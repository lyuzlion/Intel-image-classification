# ResNet50 for Intel Image Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目使用 ResNet50 模型在 Intel Image Classification 数据集上实现图像分类任务。

## 📌 项目概述
- 使用 ResNet50 模型进行学习
- 对 Intel 自然场景图像数据集（6类别）进行分类
- 包含数据增强、模型训练、准确率评估
- 最高测试准确率达 91.73%

## 🗂 数据集
**Intel Image Classification Dataset**  
- 类别：`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`
- 图像尺寸：150x150像素
- 下载地址：[Kaggle数据集](https://www.kaggle.com/puneet6060/intel-image-classification)


## ⚙️ 环境要求
- Python 3.8
- PyTorch 2.6.0
- torchvision 0.21.0
- CUDA 12.6
- 其他依赖：
```
numpy
tqdm
```

## 🚀 快速开始

### 安装依赖
```bash
git clone https://github.com/lyuzlion/Intel-image-classification.git
cd Intel-image-classification
```

### 训练模型（默认参数）
```bash
python train.py
```

## 📜 许可证
本项目基于 [MIT License](LICENSE) 授权

## 🙏 致谢
- 原始ResNet论文作者：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- 数据集提供者：[Intel Image Classification on Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification)

---
