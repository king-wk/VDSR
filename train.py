import argparse
from math import log10
import time
import os
from os import errno
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import VDSR
import random
from dataset import TrainDataset
from dataset_h5 import DatasetFromHdf5
from tqdm import tqdm
import math
 

parser = argparse.ArgumentParser(description='Train VDSR Model')
parser.add_argument('--dataset', type=str, default='DIV2K',
                    required=True, help="dataset directory name")
parser.add_argument('--patch_size', type=int, default=41,
                    help="training patch size")
parser.add_argument('--stride', type=int, default=41,
                    help="training stride")
parser.add_argument('--batch_size', type=int, default=64,
                    help="training batch size")
parser.add_argument('--epochs', type=int, default=80,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Learning Rate. Default=0.1')
parser.add_argument("--step", type=int, default=20,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--clip", type=float, default=0.4,
                    help="Clipping Gradients. Default=0.4")
parser.add_argument("--weight-decay", "--wd", default=1e-4,
                    type=float, help="Weight decay, Default: 1e-4")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4,
                    help='number of threads for data loader to use')
parser.add_argument('--gpuids', default=[0], nargs='+',
                    help='GPU ID for using')
parser.add_argument('--num_valid_image', type=int, default=10,
                    help='num_valid_image')
parser.add_argument('--model', default='VDSR', type=str, metavar='PATH',
                    help='path to test or resume model')

def get_mse(img1, img2):
    imdff = img1 - img2
    mse = math.sqrt(np.mean(imdff ** 2))
    return mse


def main():
    global args
    args = parser.parse_args()
    args.gpuids = list(map(int, args.gpuids))

    print("Args:\n", args)

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    
    cudnn.benchmark = True

    print("===> Loading datasets")
    
    # 从文件夹读取图片，处理成 patch
    dataset = TrainDataset(args.dataset, args.patch_size, args.stride, args.num_valid_image)
    # dataset = DatasetFromHdf5()

    model = VDSR()
    criterion = nn.MSELoss()
    
    if args.cuda:
        torch.cuda.set_device(args.gpuids[0])
        with torch.cuda.device(args.gpuids[0]):
            model = model.cuda()
            criterion = criterion.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_path = join("model", "{}_{}.pth".format(args.model, args.dataset))

    trained_epoch = 0
    if os.path.isfile(model_path):
        print('===>Loading checkpoint {}'.format(model_path))
        checkpoint = torch.load(model_path)
        trained_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("===>resume model (epoch={})".format(trained_epoch))
    
    print("===> Start Trainning")
    train_time = 0.0
    validate_time = 0.0
    dataset.setDatasetType('train')
    train_dataloader = DataLoader(dataset=dataset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
    for epoch in range(trained_epoch + 1, args.epochs + 1):
        start_time = time.time()
        train(model, criterion, epoch, optimizer, train_dataloader)
        elapsed_time = time.time() - start_time
        train_time += elapsed_time
        print("===> Epoch[{}-train](complete): {:.2f} seconds\n".format(epoch, elapsed_time))
        save_model(model, optimizer, epoch)
        # 每10轮验证一次
        # if epoch % 10 == 0:
        #     # save_model(model, optimizer, epoch)
        #     print("===> Start Validating")
        #     dataset.setDatasetType('valid')
        #     valid_dataloader = DataLoader(dataset=dataset, num_workers=args.threads, batch_size=args.batch_size, shuffle=False)
        #     start_time = time.time()
        #     validate(model, criterion, valid_dataloader)
        #     elapsed_time = time.time() - start_time
        #     validate_time += elapsed_time
        #     print("===> Epoch[{}-valid](complete): {:.2f} seconds\n".format(epoch, elapsed_time))
        #     dataset.setDatasetType('train')
        #     train_dataloader = DataLoader(dataset=dataset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
    
    print("===> Finish Training!")
    print("===> Average training time per epoch: {:.2f} seconds".format(train_time / (args.epochs - trained_epoch)))
    # print("===> Average validation time per epoch: {:.2f} seconds".format(validate_time / (args.epochs - trained_epoch)))
    print("===> Training time: {:.2f} seconds".format(train_time))
    # print("===> Validation time: {:.2f} seconds".format(validate_time))
    # print("===> Total time: {:.2f} seconds".format(train_time + validate_time))


def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    return lr


def train(model, criterion, epoch, optimizer, train_dataloader):
    lr = adjust_learning_rate(epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch: {}, lr: {}".format(epoch, optimizer.param_groups[0]["lr"]))

    epoch_loss = 0
    model.train()
    with tqdm(total=len(train_dataloader), desc='epoch[{}]'.format(epoch)) as pbar:
        for iteration, batch in enumerate(train_dataloader, 1):
            input, target = Variable(batch[0]), Variable(
                batch[1], requires_grad=False)
            if args.cuda:
                input = input.cuda()
                target = target.cuda()
            # print("input:{}".format(input.shape))
            # print(input[0])
            # print("target:{}".format(target.shape))
            # print(target[0])
            # print("input_target_mse:", get_mse(np.array(input.data.cpu()), np.array(target.data.cpu())))
            # assert np.sum(np.array(input.cpu() == target.cpu())) == input.shape[0] * 41 * 41, "input != target"
            # print(np.sum(np.array(input[0].cpu() == target[0].cpu())) == 41 * 41)
            # 训练的时候，随机
            if random.random() >= 0.5:
                # 50% 随机跳出
                idx = random.randint(0, 18)  # idx=0，一层都不经过，idx=18，跑完整个网络
            else:
                # 50% 概率跑完整个模型
                idx = 18
            # print("it will run {} res layers".format(idx))
            output = model(input, 18)
            # print("output:{}".format(output.shape))
            # print(output[0])
            loss = criterion(output, target)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip / lr)
            optimizer.step()
            pbar.update(1)

        # print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(train_dataloader), loss.item()))
    print("AVG Loss: {}".format(epoch_loss / len(train_dataloader)))
    # print("===> Epoch[{}-train](complete): Loss: {:.4f}".format(epoch, epoch_loss / len(train_dataloader)))


def validate(model, criterion, valid_dataloader):
    with torch.no_grad():
        model.eval()
        sr_avg_psnr = [0. for idx in range(19)]
        bicubic_avg_psnr = 0.
        for batch in valid_dataloader:
            input, target = Variable(batch[0]), Variable(batch[1])
            if args.cuda:
                input = input.cuda()
                target = target.cuda()

            bicubic_mse = criterion(input, target)
            bicubic_psnr = 10 * log10(1. / bicubic_mse.item())
            bicubic_avg_psnr += bicubic_psnr
            for idx in range(19):  # 0-18, 0 表示一个中间残差层也不经过，18 表示经过所有中间残差层
                output = model(input, idx)
                sr_mse = criterion(output, target)
                sr_psnr = 10 * log10(1. / sr_mse.item())
                sr_avg_psnr[idx] += sr_psnr
        print("===> BICUBIC_PSNR(dB):\n{:.6f}".format(bicubic_avg_psnr / len(valid_dataloader)))
        print("===> SR_PSNR(dB):")
        for idx in range(19):
            print("{:2} layer: {:.6f}".format(idx, sr_avg_psnr[idx] / len(valid_dataloader)))


def save_model(model, optimizer, epoch):
    try:
        if not(os.path.isdir('model')):
            os.makedirs(os.path.join('model'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
        
    model_path = "model/{}_{}.pth".format(args.model, args.dataset)
    state = {
            'epoch': epoch,
            'model': model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
    torch.save(state, model_path)
    print("Checkpoint saved to {}\n".format(model_path))
    model_path = "model/{}_{}_{}.pth".format(args.model, args.dataset, epoch)
    # 每10轮保存一个最后结果的模型
    if epoch % 10 == 0:
        torch.save(model, model_path)


if __name__ == '__main__':
    main()
