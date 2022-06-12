import random
import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import torchvision as vision
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, InterpolationMode
from typing import Tuple
from torch import Tensor
import glob
import h5py
from utils import convert_rgb_to_y, convert_rgb_to_ycbcr


toPIL = vision.transforms.ToPILImage()


class TrainDataset(data.Dataset):
    def __init__(self, dataset, patch_size=41, stride=41, batch_size=64, num_valid_image=10) -> None:
        super().__init__()
        self.hr_images_path = join("dataset", "{}".format(dataset), "{}_train_HR".format(dataset))
        self.lr_images_prefix = join("dataset", "{}".format(dataset), "{}_train_LR_bicubic".format(dataset))
        self.lr_patches = []
        self.hr_patches= []
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_valid_image = num_valid_image
        self.dataset_type = None  # choose between train, valid
        self.patch_num = 0
        self._setup()

    def _setup(self) -> None:
        hr_image_names = sorted(glob.glob('{}/*'.format(self.hr_images_path)))
        print("hr_images_path:{}".format(self.hr_images_path))
        print("hr_image_names:{}".format(len(hr_image_names)))
        for hr_name in hr_image_names:
            hr = Image.open(hr_name).convert('RGB')  # 高分辨率图片
            hr = np.array(hr).astype(np.float32)
            # 处理成只有 Y 通道
            hr = convert_rgb_to_y(hr)
            # crop 一个 patch
            # 这种 crop 的情况会出现最边上的没有进行超分
            for i in range(0, hr.shape[0] - self.patch_size + 1, self.stride):
                for j in range(0, hr.shape[1] - self.patch_size + 1, self.stride):
                    self.hr_patches.append(hr[i:i + self.patch_size, j:j + self.patch_size])
        # self.image_num = len(hr_image_names)
        self.patch_num = len(self.hr_patches)
        lr_patch = 0
        print("hr_patch_num: {}".format(len(self.hr_patches)))
        for scale in [2, 3, 4]:
            lr_images_path = join(self.lr_images_prefix, "X{}".format(scale))
            print("lr_images_path: {}".format(lr_images_path))
            lr_image_names = sorted(glob.glob('{}/*'.format(lr_images_path)))
            print("lr_image_names: {}".format(len(lr_image_names)))
            # assert len(lr_image_names) == self.image_num, "hr_image_num != lr_image_num"
            for lr_name in lr_image_names:
                lr = Image.open(lr_name).convert('RGB')  # 低分辨率图片
                lr = lr.resize((lr.width * scale, lr.height * scale), resample=Image.BICUBIC)  # 双三次插值处理
                lr = np.array(lr).astype(np.float32)
                # 处理成只有 Y 通道
                lr = convert_rgb_to_y(lr)
                # crop 一个 patch
                # 这种 crop 的情况会出现最边上的没有进行超分
                for i in range(0, lr.shape[0] - self.patch_size + 1, self.stride):
                    for j in range(0, lr.shape[1] - self.patch_size + 1, self.stride):
                        self.lr_patches.append(lr[i:i + self.patch_size, j:j + self.patch_size])
            print("lr_patch_num: {}".format(len(self.lr_patches) - lr_patch))
            lr_patch = len(self.lr_patches)
        assert len(self.hr_patches) * 3 == len(self.lr_patches), "hr_patches * 3 != lr_patches"
        # print("lr_patch_num: {}, hr_patch_num: {}".format(len(self.lr_patches), len(self.hr_patches)))
    
    def setDatasetType(self, dataset_type: str) -> None:
        """
        设置dataset的类型
        """
        assert dataset_type in ['train', 'valid']
        self.dataset_type = dataset_type

    def lenTrain(self) -> int:
        """ train data的长度
        """
        return len(self.lr_patches)

    def lenValid(self) -> int:
        """valid data的长度
        """
        return self.num_valid_image

    def __len__(self):
        if self.dataset_type == 'train':
            return self.lenTrain()
        elif self.dataset_type == 'valid':
            return self.lenValid()
        else:
            raise NotImplementedError
        # return len(self.lr_patches)

    def __getitem__(self, index: int):
        if self.dataset_type == 'train':
            return np.expand_dims(self.lr_patches[index] / 255., 0), np.expand_dims(self.hr_patches[index % self.patch_num] / 255., 0)
        elif self.dataset_type == 'valid':
            # 每隔 (self.lenTrain() // self.num_valid_image)
            return np.expand_dims(self.lr_patches[(self.lenTrain() // self.num_valid_image) * index] / 255., 0), np.expand_dims(self.hr_patches[((self.lenTrain() // self.num_valid_image) * index) % self.patch_num] / 255., 0)
        else:
            raise NotImplementedError


# 测试集暂时不切 patch
class TestDataset(data.Dataset):
    def __init__(self, dataset, batch_size=64) -> None:
        super().__init__()
        self.hr_images_path = join("dataset", "benchmark", "{}".format(dataset), "HR")
        self.lr_images_prefix = join("dataset", "benchmark", "{}".format(dataset), "LR_bicubic")
        self.lr_images = []
        self.hr_images= []
        self.batch_size = batch_size
        self.image_num = 0
        self._setup()

    def _setup(self):
        hr_image_names = sorted(glob.glob('{}/*'.format(self.hr_images_path)))
        print("hr_images_path:{}".format(self.hr_images_path))
        print("hr_image_names:{}".format(len(hr_image_names)))
        for hr_name in hr_image_names:
            hr = Image.open(hr_name).convert('RGB')  # 高分辨率图片
            hr = np.array(hr).astype(np.float32)
            # 保留三个通道
            hr = convert_rgb_to_ycbcr(hr)
            self.hr_images.append(hr)
        self.image_num = len(self.hr_images)
        lr_image = 0
        print("hr_image_num: {}".format(len(self.hr_images)))
        for scale in [2, 3, 4]:
            lr_images_path = join(self.lr_images_prefix, "X{}".format(scale))
            print("lr_images_path: {}".format(lr_images_path))
            lr_image_names = sorted(glob.glob('{}/*'.format(lr_images_path)))
            print("lr_image_names: {}".format(len(lr_image_names)))
            for lr_name in lr_image_names:
                lr = Image.open(lr_name).convert('RGB')  # 低分辨率图片
                lr = lr.resize((lr.width * scale, lr.height * scale), resample=Image.BICUBIC)  # 双三次插值处理
                lr = np.array(lr).astype(np.float32)
                # 保留三个通道
                lr = convert_rgb_to_ycbcr(lr)
                self.lr_images.append(lr)
            print("lr_image_num: {}".format(len(self.lr_images) - lr_image))
            lr_image = len(self.lr_images)
        assert len(self.hr_images) * 3 == len(self.lr_images), "hr_images * 3 != lr_images"
    
    def __len__(self):
        return len(self.lr_images)
    
    # 返回一个 YCbCr 三通道的图片，在计算之前，使用 Y 通道，计算完合并三通道转成 RGB，然后计算 PSNR 和 SSIM
    def __getitem__(self, index: int):
        return self.lr_images[index] / 255., self.hr_images[index % self.image_num] / 255.
