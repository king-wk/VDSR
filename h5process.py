from PIL import Image
import argparse
import math
import numpy as np
from os.path import join
import h5py
import glob
from utils import convert_rgb_to_y

# 训练的数据 patch_size=41，不重叠
# 测试的数据是一整张图片

def process_train_data(args):
    hr_images_path = join("dataset", "{}".format(args.dataset), "{}_train_HR".format(args.dataset))
    lr_images_prefix = join("dataset", "{}".format(args.dataset), "{}_train_LR_bicubic".format(args.dataset))

    lr_patches = []
    hr_patches = []
    h5file = join("dataset", "{}".format(args.dataset), "train_data.h5")
    h5_file = h5py.File(h5file, 'w')
    
    image_count = 0
    for scale in [2, 3, 4]:
        
        lr_images_path = join(lr_images_prefix, "X{}".format(scale))
        print("hr_images_path:{}, lr_images_path:{}".format(hr_images_path, lr_images_path))
        hr_image_names = sorted(glob.glob('{}/*'.format(hr_images_path)))
        lr_image_names = sorted(glob.glob('{}/*'.format(lr_images_path)))
        print("hr_image_names:{}, lr_image_names:{}".format(len(hr_image_names), len(lr_image_names)))
        for hr_name, lr_name in zip(hr_image_names, lr_image_names):
            image_count += 1
            hr = Image.open(hr_name).convert('RGB')  # 高分辨率图片
            lr = Image.open(lr_name).convert('RGB')  # 低分辨率图片
            assert lr.width * scale == hr.width and lr.height * scale == hr.height, "hr != lr * scale"
            lr = lr.resize((lr.width * scale, lr.height * scale), resample=Image.BICUBIC)  # 双三次插值处理
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            # crop 一个 patch
            # 这种 crop 的情况会出现最边上的没有进行超分
            for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
                for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                    lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                    hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])
        print("image_count:{}".format(image_count))
        lr_patches = np.array(lr_patches, dtype=object)
        hr_patches = np.array(hr_patches, dtype=object)
        h5_file.create_dataset('lr', data=lr_patches)
        h5_file.create_dataset('hr', data=hr_patches)
    
    h5_file.close()


def process_test_data_y_channel(args):
    hr_images_path = join("dataset", "{}".format(args.dataset), "{}_train_HR".format(args.dataset))
    lr_images_prefix = join("dataset", "{}".format(args.dataset), "{}_train_LR_bicubic".format(args.dataset))

    hr_images = []
    lr_images = []
    h5file = join("dataset", "{}".format(args.dataset), "train(image)_y_channel.h5")

    h5_file = h5_file = h5py.File(h5file, 'w')


def process_test_data_all_channel(args):
    hr_images_path = join("dataset", "{}".format(args.dataset), "{}_train_HR".format(args.dataset))
    lr_images_prefix = join("dataset", "{}".format(args.dataset), "{}_train_LR_bicubic".format(args.dataset))

    hr_images = []
    lr_images = []
    h5file = join("dataset", "{}".format(args.dataset), "train(image)_all_channel.h5")

    h5_file = h5_file = h5py.File(h5file, 'w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)  # dataset 名字
    parser.add_argument('--patch_size', type=int, default=41)  # patch 大小
    parser.add_argument('--stride', type=int, default=41)  # 隔多远 crop 一个 patch，VDSR 不重叠，所以默认就是 patch_size 的大小
    # parser.add_argument('--scale', type=int, default=2)  # 放大倍数
    parser.add_argument('--patch', action='store_true')  # 是否将图片变成 patch 输入进去
    parser.add_argument('--train', action='store_true')  # 处理训练数据还是测试数据
    parser.add_argument('--y_channel', action='store_true')  # 测试数据是否是单通道
    args = parser.parse_args()

    if args.train:
        process_train_data(args)
    elif args.y_channel:
        process_test_data_y_channel(args)
    else:
        process_test_data_all_channel(args)