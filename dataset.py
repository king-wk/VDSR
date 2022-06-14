import random
from turtle import width
import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import torchvision as vision
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, InterpolationMode, ToPILImage
from typing import Tuple
from torch import Tensor
import glob
from tqdm import tqdm

toPIL = vision.transforms.ToPILImage()


class TrainDataset(data.Dataset):
    def __init__(self, dataset, patch_size=41, stride=41, num_valid_image=10) -> None:
        super().__init__()
        self.dataset = dataset
        # self.hr_images_path = join("dataset", "benchmark", "{}".format(dataset), "HR")
        self.hr_images_path = join("dataset", "{}".format(dataset), "{}_test_HR".format(dataset))
        # self.lr_images_prefix = join("dataset", "benchmark", "{}".format(dataset), "LR_bicubic")
        self.lr_patches = []
        self.hr_patches= []
        self.patch_size = patch_size
        self.stride = stride
        self.num_valid_image = num_valid_image
        self.dataset_type = None  # choose between train, valid
        self.patch_num = 0
        self._setup()

    def _setup(self) -> None:
        hr_image_names = sorted(glob.glob('{}/*'.format(self.hr_images_path)))
        print("hr_images_path:{}".format(self.hr_images_path))
        print("hr_image_num:{}".format(len(hr_image_names)))
        # lr_images_path = join(self.lr_images_prefix, "X{}".format(scale))
        # print("lr_images_path: {}".format(lr_images_path))
        # lr_image_names = sorted(glob.glob('{}/*'.format(lr_images_path)))
        # print("lr_image_num: {}".format(len(lr_image_names)))
        # assert len(lr_image_names) == self.image_num, "hr_image_num != lr_image_num"
        # for hr_name, lr_name in zip(hr_image_names, lr_image_names):
        with tqdm(total=len(hr_image_names), desc='loading[{}]'.format(self.dataset)) as pbar:
            for hr_name in hr_image_names:
                # print(hr_name)
                image_patch = 0
                # hr_img 和 lr_img 是原始图片，不能改变它们
                img = Image.open(hr_name).convert('YCbCr')  # 高分辨率图片
                for scale in [2, 3, 4]:
                    # print("scale:{}".format(scale))
                    for flip in range(3):  # 翻转： 0:不翻转 1:左右翻转 2:上下翻转
                        for degree in range(4):  # 旋转角度： degree * 90 (0, 90, 180, 270)
                            # lr_img = Image.open(lr_name).convert('YCbCr')  # 低分辨率图片
                            # 数据增强
                            if flip == 1:
                                hr_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                                # lr = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
                                # print("左右翻转")
                            elif flip == 2:
                                hr_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                                # lr = lr_img.transpose(Image.FLIP_TOP_BOTTOM)
                                # print("上下翻转")
                            else:
                                hr_img = img
                                # lr = lr_img
                                # print("不翻转")
                            
                            hr_img = hr_img.rotate(degree * 90)
                            # lr = lr.rotate(degree * 90)
                            # print("旋转{}°".format(degree * 90))
                            width = hr_img.width - hr_img.width % scale
                            height = hr_img.height - hr_img.height % scale
                            # 要根据 scale 修改 hr 的 width 和 height，可能不能整除
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
                            # 处理成只有 Y 通道
                            hr = hr[0,:,:]
                            lr = lr[0,:,:]
                            assert hr.shape[0] == lr.shape[0] and hr.shape[1] == lr.shape[1], "hr.shape({}x{}) != lr.shape({}x{})".format(hr.shape[0], hr.shape[1], lr.shape[0], lr.shape[1])
                            # 随机 crop 一个 patch
                            height_ = random.randrange(0, hr.shape[0] - self.patch_size + 1)
                            width_ = random.randrange(0, hr.shape[1] - self.patch_size + 1)
                            hr = hr[height_:height_ + self.patch_size, width_:width_ + self.patch_size]
                            lr = lr[height_:height_ + self.patch_size, width_:width_ + self.patch_size]
                            # print("hr:{}, lr:{}".format(hr.shape, lr.shape))
                            self.hr_patches.append(hr)
                            self.lr_patches.append(lr)
                            image_patch += 1
                            # 不随机,网格切 patch
                            # 这种 crop 的情况会出现最边上的没有进行超分
                            # for i in range(0, hr.shape[0] - self.patch_size + 1, self.stride):
                            #     for j in range(0, hr.shape[1] - self.patch_size + 1, self.stride):
                            #         self.hr_patches.append(hr[i:i + self.patch_size, j:j + self.patch_size])
                            #         self.lr_patches.append(lr[i:i + self.patch_size, j:j + self.patch_size])
                            #         image_patch += 1
                    # print("this image have {} patches.".format(image_patch))
                pbar.update(1)
        self.patch_num = len(self.hr_patches)
        print("total patch_num: {}".format(self.patch_num))
        assert len(self.hr_patches) == len(self.lr_patches), "hr_patches({}) != lr_patches({})".format(len(self.lr_patches), len(self.hr_patches))
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
        return self.patch_num

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

    def __getitem__(self, index: int):
        if self.dataset_type == 'train':
            return np.expand_dims(self.lr_patches[index], 0), np.expand_dims(self.hr_patches[index], 0)
        elif self.dataset_type == 'valid':
            # 每隔 (self.lenTrain() // self.num_valid_image)
            return np.expand_dims(self.lr_patches[(self.lenTrain() // self.num_valid_image) * index], 0), np.expand_dims(self.hr_patches[(self.lenTrain() // self.num_valid_image) * index], 0)
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
        self.input_transform = Compose([
            ToTensor(),
        ])
        self.target_transform = Compose([
            ToTensor(),
        ])
        self._setup()

    def _setup(self):
        hr_image_names = sorted(glob.glob('{}/*'.format(self.hr_images_path)))
        print("hr_images_path:{}".format(self.hr_images_path))
        for hr_name in hr_image_names:
            hr = Image.open(hr_name).convert('YCbCr')  # 高分辨率图片
            hr = self.target_transform(hr)
            hr = np.array(hr).astype(np.float32)
            # 保留三个通道
            self.hr_images.append(hr)
        self.image_num = len(self.hr_images)
        lr_image = 0
        print("hr_image_num: {}".format(len(self.hr_images)))
        for scale in [2, 3, 4]:
            lr_images_path = join(self.lr_images_prefix, "X{}".format(scale))
            print("lr_images_path: {}".format(lr_images_path))
            lr_image_names = sorted(glob.glob('{}/*'.format(lr_images_path)))
            for lr_name in lr_image_names:
                lr = Image.open(lr_name).convert('YCbCr')  # 低分辨率图片
                lr = lr.resize((lr.width * scale, lr.height * scale), resample=Image.BICUBIC)  # 双三次插值处理
                lr = self.input_transform(lr)
                lr = np.array(lr).astype(np.float32)
                # 保留三个通道
                self.lr_images.append(lr)
            print("lr_image_num: {}".format(len(self.lr_images) - lr_image))
            lr_image = len(self.lr_images)
        assert len(self.hr_images) * 3 == len(self.lr_images), "hr_images * 3 != lr_images"
    
    def __len__(self):
        return len(self.lr_images)
    
    # 返回一个 YCbCr 三通道的图片，在计算之前，使用 Y 通道，计算完合并三通道转成 RGB，然后计算 PSNR 和 SSIM
    def __getitem__(self, index: int):
        return self.lr_images[index], self.hr_images[index % self.image_num]
