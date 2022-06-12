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
from utils import convert_rgb_to_y


toPIL = vision.transforms.ToPILImage()


# crop_size 是输出网络的大小，如果不能整除 upscale_factor
# def calculate_valid_crop_size(crop_size, upscale_factor):
#     return crop_size - (crop_size % upscale_factor)


def get_input_transform(height, width, upscale_factor):
    return Compose([
        CenterCrop((height, width)),  # 将图片裁剪成 height x width
        Resize((height // upscale_factor, width // upscale_factor)),  # 下采样
        Resize((height, width), interpolation=InterpolationMode.BICUBIC),  # 双三次插值
        ToTensor(),
    ])


def get_target_transform(height, width):
    return Compose([
        CenterCrop((height, width)),  # 将图片裁剪成 height x width（目标图片）
        ToTensor(),
    ])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    # 只需要 Y 通道
    return img


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


# 直接从某个高分辨率图片的文件夹读取图片，然后将图片下采样得到低分辨率图片，然后插值得到 input
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, height=540, width=960, patch_size=64, num_batch=64, num_update_per_epoch=200, num_valid_image=10):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.dataset_type = None  # choose between train, valid, test
        self.height = height  # 高分辨率图片的宽和高
        self.width = width
        self.patch_size = patch_size  # 输入网络的 patch 大小
        self.input_transform = Compose([
            ToTensor(),
        ])
        self.target_transform = Compose([
            ToTensor(),
        ])
        self.num_batch = num_batch  # 默认是 64 epoch
        self.num_update_per_epoch = num_update_per_epoch  # todo，默认 200
        self.num_valid_image = num_valid_image  # 用于验证的图片数量
        self.input_image = []  # 存放低分辨率图片插值之后的图片
        self.target_image = []
        self._setup()
    
    def _setup(self) -> None:
        target_transform = get_target_transform(self.height, self.width)
        for index in range(len(self.image_filenames)):
            img = Image.open(self.image_filenames[index]).convert('YCbCr')
            # 因为网络是 x2 x3 x4 混在一起，随机下采样倍数
            upscale_factor = random.randint(2, 4)
            target = target_transform(img)
            input_transform = get_input_transform(self.height, self.width, upscale_factor)
            input = input_transform(img)
            # print("input_shape:{}".format(np.array(input).shape))
            self.input_image.append(input)
            self.target_image.append(target)

    def setDatasetType(self, dataset_type: str) -> None:
        """
        设置dataset的类型
        """
        assert dataset_type in ['train', 'valid', 'test']
        self.dataset_type = dataset_type

    def lenTrain(self) -> int:
        """ train data的长度
        @description  :
        ---------
        traindata的长度: num_batch*num_update_per_epoch
        """
        return self.num_batch * self.num_update_per_epoch

    def lenValid(self) -> int:
        """valid data的长度
        @description  :
        ---------
        num_valid_image
        """
        return self.num_valid_image

    def lenTest(self) -> int:
        """test data的长度
        @description  :
        ---------
        所有帧的数目
        """
        return len(self.target_image)

    def __len__(self):
        if self.dataset_type == 'train':
            return self.lenTrain()
        elif self.dataset_type == 'valid':
            return self.lenValid()
        elif self.dataset_type == 'test':
            return self.lenTest()
        else:
            raise NotImplementedError

    def getItemTrain(self) -> Tuple[Tensor, ...]:
        frame_idx = random.randrange(0, len(self.target_image))
        input = self.input_image[frame_idx]
        target = self.target_image[frame_idx]
        height_ = random.randrange(0, self.height - self.patch_size + 1)
        width_ = random.randrange(0, self.width - self.patch_size + 1)

        input = toPIL(input).crop((width_, height_, width_ + self.patch_size, height_ + self.patch_size))
        target = toPIL(target).crop((width_, height_, width_ + self.patch_size, height_ + self.patch_size))
        input = self.input_transform(np.array(input)[:,:,0])
        target = self.input_transform(np.array(target)[:,:,0])
        # print("train input_shape:{}".format(np.array(input).shape))
        return input, target

    def getItemTest(self, index: int) -> Tuple[Tensor, ...]:
        """
        依据index找到对应的lr图片,hr图片,ffmpeg插值后的图片
        Args:
            index:图片的索引值

        Returns:
            input, target, upscaled
        """
        input = self.input_image[index]
        target = self.target_image[index]
        
        # crop 一个 patch_size x patch_size 大小的 patch 输入网络
        # height_ = random.randrange(0, self.height - self.patch_size + 1)
        # width_ = random.randrange(0, self.width - self.patch_size + 1)
        # print("test input_shape:{}".format(np.array(input).shape))

        # @TODO: 加载的图片一会是 3x128x128, 一会是 128x128x3
        # totensor 之后是 c x H x W
        # 使用 toPIL 之后是 H x W x c

        # input = toPIL(input).crop((width_, height_, width_ + self.patch_size, height_ + self.patch_size))
        # target = toPIL(target).crop((width_, height_, width_ + self.patch_size, height_ + self.patch_size))
        
        
        # 只提取一个通道输入网络
        # input = self.input_transform(np.array(input)[:,:,0])
        # target = self.input_transform(np.array(target)[:,:,0])

        # 将整张图片输入时使用这两行
        input = self.input_transform(np.array(input)[0,:,:])
        target = self.input_transform(np.array(target)[0,:,:])

        # print("test input_shape:{}".format(np.array(input).shape))
        return input, target

    def __getitem__(self, index: int):
        if self.dataset_type == 'train':
            return self.getItemTrain()
        elif self.dataset_type == 'valid':
            # 每隔(self.lenTest() // self.train_param.num_valid_image)
            return self.getItemTest((self.lenTest() // self.num_valid_image) * index)
        elif self.dataset_type == 'test':
            return self.getItemTest(index)
        else:
            raise NotImplementedError


