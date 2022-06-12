# VDSR

Use VDSR train general SR model([VDSR: Accurate Image Super-Resolution Using Very Deep Convolutional Networks(CVPR2016)](http://cv.snu.ac.kr/research/VDSR/))

## 参数
- 网络深度：20 (中间残差层18层，训练时，50% 概率从中间某一层随机跳出，50% 概率通过所有的层)
- patch_size：和感受野一致 41 * 41
- batch size：64
- momentum：0.9
- weight decay parameter：0.0001
- epoch：80
- 初始学习率 0.1，每 20 个 epoch 衰减 10 倍

## 数据集
对于输入网络的低分辨率图像需要先进行 bicubic 双三次插值到对应高分辨率图像大小，然后只需要将 Y 通道进行 patch 划分保存。

### 训练数据集
- DIV2K：训练集 900 张图片，一共 4386186 个 patch，每轮 68535 iteration，验证集取其中 10 个 patch，每 10 轮验证一次，并计算在每一层跳出的增益，即 PSNR

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
