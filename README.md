# VDSR

Use VDSR train general SR model

[VDSR: Accurate Image Super-Resolution Using Very Deep Convolutional Networks(CVPR2016)](http://cv.snu.ac.kr/research/VDSR/)

## 参数
- 网络深度：20 (中间残差层 18 层，训练时，50% 概率从中间某一层随机跳出，50% 概率通过所有的层)
- patch_size：和感受野一致 41 * 41
- batch size：64
- momentum：0.9
- weight decay parameter：0.0001
- epoch：80
- 初始学习率 0.1，每 20 个 epoch 衰减 10 倍
- clip：梯度裁减，论文未确定使用值，暂时使用 0.4，根据论文，每一轮的梯度裁减使用 clip / lr

## 数据集
对于输入网络的低分辨率图像需要先进行 bicubic 双三次插值到对应高分辨率图像大小，然后只需要将 Y 通道进行 patch 划分保存。

### 训练数据集
- DIV2K：800 张图片
- scale：x2 x3 x4 混合训练，因此一共会有 3 * 800 张图片
- 数据增强：12 种方式(旋转：0°，90°，180°，270°；翻转：不翻转，上下翻转，左右翻转)，因此一共会有 12 * 3 * 800 = 28800 张图片
- 划分 patch：如果随机划分，一张图片一个 patch；如果网格划分，图片不重叠划分
- 验证集：取其中 10 个 patch，每 10 轮验证一次，并计算在每一层跳出的增益，即 PSNR

### 测试数据集
- DIV2K：100 张图片
- B100：100 张图片
- Set5：5 张图片
- Set14：14 张图片
- Urban100：100 张图片

## 指标
- PSNR(三通道)
- PSNR(Y通道)
- SSIM
- MSE
