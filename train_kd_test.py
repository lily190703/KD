# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 09:49:07 2022

@author: lily
"""
# ACN
import argparse
import os

import shutil  # save_checkpoint() need
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
from callbacks import AverageMeter, Logger
from data_utils.data_loader_frames import VideoFolder  # 加载sth数据集
from model.model_lib_stu_wBk import VideoModelCoord as VideoModel  # student_model配置
from utils import save_results
from ipdb import set_trace  # 调试函数

parser = argparse.ArgumentParser(description='PyTorch Smth-Else')

# Path related arguments

parser.add_argument('--model', default='coord')
parser.add_argument('--root_frames', help='path to the folder with frames')
parser.add_argument('--json_data_train', help='path to the json file with train video meta data')
parser.add_argument('--json_data_val', help='path to the json file with validation video meta data')
parser.add_argument('--json_file_labels', help='path to the json file with ground truth labels')
parser.add_argument('--tracked_boxes', help='choose tracked boxes')
parser.add_argument('--save_path', type=str, help='frame and list save path')
parser.add_argument('--model_path', help='path to pth model')

parser.add_argument('--img_feature_dim', default=256, type=int, metavar='N',
                    help='intermediate feature dimension for image-based features')
parser.add_argument('--coord_feature_dim', default=256, type=int, metavar='N',
                    help='intermediate feature dimension for coord-based features')
parser.add_argument('--clip_gradient', '-cg', default=5, type=float,
                    metavar='W', help='gradient norm clipping (default: 5)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--size', default=224, type=int, metavar='N',
                    help='primary image input size')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '-b', default=72, type=int,
                    metavar='N', help='mini-batch size (default: 72)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print_freq', '-p', default=430, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--log_freq', '-l', default=10, type=int,
                    metavar='N', help='frequency to write in tensorboard (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num_classes', default=174, type=int,
                    help='num of class in the model')
parser.add_argument('--num_boxes', default=4, type=int,
                    help='num of boxes for each image')
parser.add_argument('--num_frames', default=16, type=int,
                    help='num of frames for the model')
parser.add_argument('--dataset', default='smth_smth',
                    help='which dataset to train')
parser.add_argument('--logdir', default='./logs',
                    help='folder to output tensorboard logs')
parser.add_argument('--logname', default='exp',
                    help='name of the experiment for checkpoints and logs')
parser.add_argument('--ckpt', default='./ckpt',
                    help='folder to output checkpoints')
parser.add_argument('--fine_tune', help='path with ckpt to restore')
parser.add_argument('--shot', default=5)
parser.add_argument('--restore_i3d')
parser.add_argument('--arch', default='ck')
parser.add_argument('--restore_custom')

best_loss = 1000000


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def adjust_learning_rate(optimzer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


def load_checkpoint(teacher_checkpoint, teacher_model):
    assert os.path.isfile(teacher_checkpoint), "No checkpoint found at '{}'".format(teacher_checkpoint)
    print("=> loading checkpoint '{}'".format(teacher_checkpoint))
    checkpoint = torch.load(teacher_checkpoint)
    if args.start_epoch is None:
        args.start_epoch = checkpoint['epoch']
    teacher_model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(teacher_checkpoint, checkpoint['epoch']))


def validate(val_loader, model, criterion, isval, epoch=None, tb_logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    logits_matrix = []
    targets_list = []
    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (box_tensors, video_label) in enumerate(val_loader):

        video_label_cuda = video_label.cuda()
        batchsize = box_tensors.shape[0]
        flag = [0] * 48

        with torch.no_grad():
            cls, label, flag = model(box_tensors, video_label_cuda, flag, isval)

            cls = cls.view((-1, 174))
            loss = criterion(cls, label)
            acc1, acc5 = accuracy(cls.cpu(), label.cpu(), topk=(1, 5))

        # measure accuracy and record loss
        losses.update(loss.item(), batchsize)
        acc_top1.update(acc1.item(), batchsize)
        acc_top5.update(acc5.item(), batchsize)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i + 1 == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss_e {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc1 {acc_top1.val:.1f} ({acc_top1.avg:.1f})\t'
                  'Acc5 {acc_top5.val:.1f} ({acc_top5.avg:.1f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, acc_top1=acc_top1, acc_top5=acc_top5))

    if args.evaluate:
        logits_matrix = np.concatenate(logits_matrix)
        targets_list = np.concatenate(targets_list)

    save_results(logits_matrix, targets_list, args, epoch)

    if epoch is not None and tb_logger is not None:
        logs = OrderedDict()
        logs['Val_EpochLoss'] = losses.avg
        logs['Val_EpochAcc@1'] = acc_top1.avg
        logs['Val_EpochAcc@5'] = acc_top5.avg
        # how many iterations we have trained
        for key, value in logs.items():
            tb_logger.log_scalar(value, key, epoch + 1)

        tb_logger.flush()

    return losses.avg


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()

    # create model
    model = VideoModel(args)

    # optionally resume from a checkpoint
    if args.start_epoch is None:
        args.start_epoch = 0

    model = model.cuda()
    # model = torch.nn.DataParallel(model)
    cudnn.benchmark = True  # benchmark模式提升计算速度，但是会使网络前馈结果略有差异

    # create validation dataset from sthv2 dataset
    dataset_val = VideoFolder(root=args.root_frames,
                              num_boxes=args.num_boxes,
                              file_input=args.json_data_val,
                              file_labels=args.json_file_labels,
                              args=args,
                              is_val=True,
                              if_augment=True,
                              model=args.model,
                              save_path=args.save_path
                              )

    # create validation loader
    val_loader = torch.utils.data.DataLoader(
        dataset_val, drop_last=True,
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=False
    )
    optimizer = torch.optim.SGD(model.parameters(), momentum=args.momentum,
                                lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # use GPU if available
    args.cuda = torch.cuda.is_available()

    model_checkpoint = args.model_path
    assert os.path.isfile(model_checkpoint), "No checkpoint found at '{}'".format(model_checkpoint)
    print("=> loading checkpoint '{}'".format(model_checkpoint))
    checkpoint = torch.load(model_checkpoint)
    if args.start_epoch is None:
        args.start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_checkpoint, checkpoint['epoch']))

    # training, start a logger
    tb_logdir = os.path.join(args.logdir, args.logname)
    if not (os.path.exists(tb_logdir)):
        os.makedirs(tb_logdir)
    tb_logger = Logger(tb_logdir)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)  # 学习率调整
        isval = True

        loss = validate(val_loader, model, criterion, isval, epoch=epoch, tb_logger=tb_logger)
