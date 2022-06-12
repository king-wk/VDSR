import argparse
from math import log10
from statistics import mode
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
import math
from dataset import DatasetFromFolder, TrainDataset
from tqdm import tqdm
 



# 输入是固定的
# 固定输出大小
parser = argparse.ArgumentParser(description='PyTorch VDSR')
parser.add_argument('--dataset', type=str, default='BSDS300',
                    required=True, help="dataset directory name")
# parser.add_argument('--height', type=int, default=540, help="network input height")
# parser.add_argument('--width', type=int, default=960, help="network input width")
# parser.add_argument('--crop_size', type=int, default=256,
#                     required=True, help="network input size")
# parser.add_argument('--upscale_factor', type=int, default=2,
#                     required=True, help="super resolution upscale factor")
parser.add_argument('--patch_size', type=int, default=41,
                    help="training patch size")
parser.add_argument('--stride', type=int, default=41,
                    help="training stride")
parser.add_argument('--batch_size', type=int, default=64,
                    help="training batch size")
parser.add_argument('--test_batch_size', type=int,
                    default=32, help="testing batch size")
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning Rate. Default=0.001')
parser.add_argument("--step", type=int, default=20,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--clip", type=float, default=0.4,
                    help="Clipping Gradients. Default=0.4")
parser.add_argument("--weight-decay", "--wd", default=1e-4,
                    type=float, help="Weight decay, Default: 1e-4")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=2,
                    help='number of threads for data loader to use')
parser.add_argument('--gpuids', default=[0], nargs='+',
                    help='GPU ID for using')
# parser.add_argument('--num_update_per_epoch', type=int, default=50,
#                     help='num_update_per_epoch')
parser.add_argument('--num_valid_image', type=int, default=10,
                    help='num_valid_image')
parser.add_argument('--model', default='VDSR', type=str, metavar='PATH',
                    help='path to test or resume model')


def main():
    global args
    args = parser.parse_args()
    args.gpuids = list(map(int, args.gpuids))

    print(args)

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    
    cudnn.benchmark = True

    print("===> Loading datasets")
    
    # 从文件夹读取图片，处理成 patch
    dataset = TrainDataset(args.dataset, args.patch_size, args.stride, args.batch_size, args.num_valid_image)

    model = VDSR()
    criterion = nn.MSELoss()
    
    if args.cuda:
        torch.cuda.set_device(args.gpuids[0])
        with torch.cuda.device(args.gpuids[0]):
            model = model.cuda()
            criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=args.gpuids, output_device=args.gpuids[0])

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model_path = join("model", "{}_{}.pth".format(args.model, args.dataset))

    trained_epoch = 0
    if os.path.isfile(model_path):
        print('===>Loading checkpoint {}'.format(model_path))
        checkpoint = torch.load(model_path)
        trained_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("===>resume model (epoch={})".format(trained_epoch))
    
    print("===> start trainning")
    train_time = 0.0
    validate_time = 0.0
    dataset.setDatasetType('train')
    train_dataloader = DataLoader(dataset=dataset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
    for epoch in range(trained_epoch + 1, args.epochs + 1):
        start_time = time.time()
        train(model, criterion, epoch, optimizer, train_dataloader)
        elapsed_time = time.time() - start_time
        train_time += elapsed_time
        print("===> {:.2f} seconds to train this epoch\n".format(elapsed_time))
        save_model(model, optimizer, epoch)
        # 每10轮验证一次
        if epoch % 10 == 0:
            # save_model(model, optimizer, epoch)
            print("===> Loading valid datasets")
            dataset.setDatasetType('valid')
            valid_dataloader = DataLoader(dataset=dataset, num_workers=args.threads, batch_size=args.batch_size, shuffle=False)
            start_time = time.time()
            validate(model, criterion, valid_dataloader)
            elapsed_time = time.time() - start_time
            validate_time += elapsed_time
            print("===> {:.2f} seconds to validate this epoch\n".format(elapsed_time))
            dataset.setDatasetType('train')
            train_dataloader = DataLoader(dataset=dataset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
    
    print("===> average training time per epoch: {:.2f} seconds".format(train_time/args.epochs))
    print("===> average validation time per epoch: {:.2f} seconds".format(validate_time/args.epochs))
    print("===> training time: {:.2f} seconds".format(train_time))
    print("===> validation time: {:.2f} seconds".format(validate_time))
    print("===> total training time: {:.2f} seconds".format(train_time + validate_time))


def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    return lr


def train(model, criterion, epoch, optimizer, train_dataloader):
    lr = adjust_learning_rate(epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    epoch_loss = 0
    with tqdm(total=100, desc='epoch[{}]'.format(epoch)) as pbar:
        for iteration, batch in enumerate(train_dataloader, 1):
            input, target = Variable(batch[0]), Variable(
                batch[1], requires_grad=False)
            # print("input:{}, target:{}".format(input.shape, target.shape))
            if args.cuda:
                input = input.cuda()
                target = target.cuda()
            print("input:{}".format(input.shape), "target:{}".format(target.shape))
            output = model(input)
            loss = criterion(output * 255., target * 255.)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip/lr)
            optimizer.step()
            pbar.update(100 / len(train_dataloader))

        # print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(train_dataloader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_dataloader)))


def validate(model, criterion, valid_dataloader):
    sr_avg_psnr = 0.
    bicubic_avg_psnr = 0.
    for batch in valid_dataloader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            input = input.cuda()
            target = target.cuda()

        bicubic_mse = criterion(input * 255., target * 255.)
        output = model(input)
        sr_mse = criterion(output * 255., target * 255.)
        sr_psnr = 10 * log10(255. * 255. / sr_mse.item())
        bicubic_psnr = 10 * log10(255. * 255. / bicubic_mse.item())
        sr_avg_psnr += sr_psnr
        bicubic_avg_psnr += bicubic_psnr
    print("===> SR_PSNR: {:.4f} dB, BICUBIC_PSNR: {:.4f}".format(sr_avg_psnr / len(valid_dataloader), bicubic_avg_psnr / len(valid_dataloader)))


def test(model, criterion, test_dataloader):
    sr_avg_psnr = 0.
    bicubic_avg_psnr = 0.
    test_time = 0.
    frame_count = 0
    for batch in test_dataloader:
        print("batch:{}".format(batch.shape))
        input = batch
        # bicubic_mse = criterion(input * 255., target * 255.)
        # input, target = Variable(batch[0]), Variable(batch[1])
        # if args.cuda:
        #     input = input.cuda()
        #     target = target.cuda()
        print("input:{}".format(input.shape))
        start_time = time.time()
        patch_num = 0
        for idx_h in range(math.ceil(args.height / args.patch_size)):  # 向上取整
            h1 = idx_h * args.patch_size  # 左边不会有问题
            h2 = args.height if idx_h == math.ceil(args.height / args.patch_size) - 1 else (idx_h + 1) * args.patch_size
            for idx_w in range(math.ceil(args.width / args.patch_size)):  # 向上取整
                # SR
                w1 =  idx_w * args.patch_size
                w2 = args.width if idx_w == math.ceil(args.width / args.patch_size) - 1 else (idx_w + 1) * args.patch_size
                patch_num += 1
                input_patch = input[:,:,h1:h2,w1:w2]
                input_patch = Variable(input_patch)
                if args.cuda:
                    input_patch = input_patch.cuda()
                input[:,:,h1:h2,w1:w2] = model(input_patch)
        print("patch_num:{}".format(patch_num))
        test_time += time.time() - start_time
        frame_count += len(input)
        # sr_mse = criterion(input * 255., target * 255.)
        # sr_psnr = 10 * log10(255. * 255. / sr_mse.item())
        # bicubic_psnr = 10 * log10(255. * 255. / bicubic_mse.item())
        # sr_avg_psnr += sr_psnr
        # bicubic_avg_psnr += bicubic_psnr
        print("frame_count:{}".format(frame_count))
    print("spent {:.4f}ms to restruction a frame.".format(test_time / frame_count * 1000.0))
    # print("===> SR_PSNR: {:.4f} dB, BICUBIC_PSNR: {:.4f}".format(sr_avg_psnr / len(test_dataloader), bicubic_avg_psnr / len(test_dataloader)))


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
    print("Checkpoint saved to {}".format(model_path))
    model_path = "model/{}_{}_{}.pth".format(args.model, args.dataset, epoch)
    # 每10轮保存一个最后结果的模型
    if epoch % 10 == 0:
        torch.save(model, model_path)


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print("===> total time: {:.2f} seconds".format(elapsed_time))