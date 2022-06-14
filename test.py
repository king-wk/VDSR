
from os.path import join
from tkinter import N
from PIL import Image
import numpy as np
import torchvision as vision
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, InterpolationMode, ToPILImage
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import glob
import math
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import VDSR
from utils import convert_ycbcr_to_rgb
 

def PSNR(img1, img2):
    imdff = img1 - img2
    mse = math.sqrt(np.mean(imdff ** 2))
    if mse == 0:
        return 100 * 100
    return 10 * math.log10(255.0 * 255.0 / mse)

image_path = "/home/netlab/Documents/VDSR/dataset/benchmark/B100/HR"
output_path = "/home/netlab/Documents/VDSR/dataset/output/B100"
model_path = "/home/netlab/Documents/VDSR/model/VDSR_Set5_80.pth"
model = VDSR()
model = torch.load(model_path)
image_names = sorted(glob.glob('{}/*'.format(image_path)))
for image_name in image_names:
    hr_img = Image.open(image_name).convert('YCbCr')  # 高分辨率图片
    name = image_name.split('/')
    name = name[-1].split('.')[0]
    for scale in [2, 3, 4]:
        width = hr_img.width - hr_img.width % scale
        height = hr_img.height - hr_img.height % scale
        target_transform = Compose([
            CenterCrop((height, width)),
            ToTensor(),
        ])
        input_transform = Compose([
            CenterCrop((height, width)),
            Resize((height // scale, width // scale), interpolation=InterpolationMode.BICUBIC),  # 下采样
            Resize((height, width), interpolation=InterpolationMode.BICUBIC),  # 双三次插值
            ToTensor(),
        ])
        hr = target_transform(hr_img)
        lr = input_transform(hr_img)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        target = convert_ycbcr_to_rgb(hr * 255.)
        bicubic = convert_ycbcr_to_rgb(lr * 255.)
        bicubic_psnr = PSNR(bicubic, target)
        target_img = Image.fromarray(np.uint8(target)).convert("RGB")
        bicubic_img = Image.fromarray(np.uint8(bicubic)).convert("RGB")
        bicubic_img.save(join(output_path, "{}_bicubic_{}.png".format(name, scale)))
        target_img.save(join(output_path, "{}_target_{}.png".format(name, scale)))

        # 处理成只有 Y 通道
        lr_y = lr[0,:,:]
        assert hr.shape[1] == lr.shape[1] and hr.shape[2] == lr.shape[2], "hr.shape({}x{}) != lr.shape({}x{})".format(hr.shape[1], hr.shape[1], lr.shape[1], lr.shape[2])
        input = Variable(torch.from_numpy(lr_y).float()).view(1, -1, lr_y.shape[0], lr_y.shape[1])
        model = model.cuda()
        input = input.cuda()
        output = model(input, 18)
        sr_y = output.data[0].cpu().numpy().astype(np.float32)
        lr[0,:,:] = sr_y

        sr = convert_ycbcr_to_rgb(lr * 255.)
        sr_psnr = PSNR(sr, target)
        sr_img = Image.fromarray(np.uint8(sr)).convert("RGB")
        sr_img.save(join(output_path, "{}_output_{}.png".format(name, scale)))
        print("image {}({}x{}) x{}({}x{}):".format(name, height // scale, width // scale, scale, height, width))
        print("BICUBIC PSNR: {:4f}, SR PSNR: {:4f}\n".format(bicubic_psnr, sr_psnr))
