# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Tue Nov 5 11:56:06 2020
@author: hongwang (hongwang9209@hotmail.com)
MICCAI2021: ``InDuDoNet: An Interpretable Dual Domain Network for CT Metal Artifact Reduction''
paper link： https://arxiv.org/pdf/2109.05298.pdf
"""
from __future__ import print_function
import argparse
import os
import torch
import torch.nn.functional as  F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from math import ceil
from deeplesion.Dataset import MARTrainDataset
from network.indudonet import InDuDoNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./deep_lesion/", help='txt path to training spa-data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--patchSize', type=int, default=416, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='total number of training epochs')
parser.add_argument('--num_channel', type=int, default=32, help='the number of dual channels')  # refer to https://github.com/hongwang01/RCDNet for the channel concatenation strategy
parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='the number of total iterative stages')
parser.add_argument('--resume', type=int, default=0, help='continue to train')
parser.add_argument("--milestone", type=int, default=[40, 80], help="When to decay learning rate")
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--log_dir', default='./logs/', help='tensorboard logs')
parser.add_argument('--model_dir', default='./models/', help='saving model')
parser.add_argument('--eta1', type=float, default=1, help='initialization for stepsize eta1')
parser.add_argument('--eta2', type=float, default=5, help='initialization for stepsize eta2')
parser.add_argument('--alpha', type=float, default=0.5, help='initialization for weight factor')
parser.add_argument('--gamma', type=float, default=1e-1, help='hyper-parameter for balancing different loss items')
opt = parser.parse_args()

# create path

try:
    os.makedirs(opt.log_dir)
except OSError:
    pass
try:
    os.makedirs(opt.model_dir)
except OSError:
    pass
cudnn.benchmark = True


def train_model(net,optimizer, scheduler,datasets):
    data_loader = DataLoader(datasets, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers),
                             pin_memory=True)
    num_data = len(datasets)
    num_iter_epoch = ceil(num_data / opt.batchSize)
    writer = SummaryWriter(opt.log_dir)
    step = 0
    for epoch in range(opt.resume, opt.niter):
        mse_per_epoch = 0
        tic = time.time()
        # train stage
        lr = optimizer.param_groups[0]['lr']
        phase = 'train'
        for ii, data in enumerate(data_loader):
            Xma, XLI, Xgt, mask, Sma, SLI, Sgt, Tr = [x.cuda() for x in data]
            net.train()
            optimizer.zero_grad()
            ListX, ListS, ListYS= net(Xma, XLI, mask, Sma, SLI, Tr)
            loss_l2YSmid = 0.1 * F.mse_loss(ListYS[opt.S -2], Sgt)
            loss_l2Xmid = 0.1 * F.mse_loss(ListX[opt.S -2] * (1 - mask), Xgt * (1 - mask))
            loss_l2YSf = F.mse_loss(ListYS[-1], Sgt)
            loss_l2Xf = F.mse_loss(ListX[-1] * (1 - mask), Xgt * (1 - mask))
            loss_l2YS = loss_l2YSf + loss_l2YSmid
            loss_l2X = loss_l2Xf +  loss_l2Xmid
            loss = opt.gamma * loss_l2YS + loss_l2X
            loss.backward()
            optimizer.step()
            mse_iter = loss.item()
            mse_per_epoch += mse_iter
            if ii % 400 == 0:
                template = '[Epoch:{:>2d}/{:<2d}] {:0>5d}/{:0>5d}, Loss={:5.2e},  Lossl2YS={:5.2e}, Lossl2X={:5.2e}, lr={:.2e}'
                print(template.format(epoch + 1, opt.niter, ii, num_iter_epoch, mse_iter, loss_l2YS, loss_l2X, lr))
            writer.add_scalar('Loss', loss, step)
            writer.add_scalar('Loss_YS', loss_l2YS, step)
            writer.add_scalar('Loss_X', loss_l2X, step)
            step += 1
        mse_per_epoch /= (ii + 1)
        print('Loss={:+.2e}'.format(mse_per_epoch))
        print('-' * 100)
        scheduler.step()
        # save model
        torch.save(net.state_dict(), os.path.join(opt.model_dir, 'InDuDoNet_latest.pt'))
        if epoch % 10 == 0:
            # save model
            model_prefix = 'model_'
            save_path_model = os.path.join(opt.model_dir, model_prefix + str(epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'step': step + 1,
            }, save_path_model)
            torch.save(net.state_dict(), os.path.join(opt.model_dir, 'InDuDoNet_%d.pt' % (epoch + 1)))
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc - tic))
    writer.close()
    print('Reach the maximal epochs! Finish training')

if __name__ == '__main__':
    def print_network(name, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('name={:s}, Total number={:d}'.format(name, num_params))
    net = InDuDoNet(opt).cuda()
    print_network("InDuDoNet:", net)
    optimizer= optim.Adam(net.parameters(), betas=(0.5, 0.999), lr=opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestone,gamma=0.5)  # learning rates
    # from opt.resume continue to train
    for _ in range(opt.resume):
        scheduler.step()
    if opt.resume:
        net.load_state_dict(torch.load(os.path.join(opt.model_dir, 'InDuDoNet_%d.pt' % (opt.resume))))
        print('loaded checkpoints, epoch{:d}'.format(opt.resume))
    # load dataset
    train_mask = np.load(os.path.join(opt.data_path, 'trainmask.npy'))
    train_dataset = MARTrainDataset(opt.data_path, opt.patchSize, train_mask)

    # train model
    train_model(net, optimizer, scheduler,train_dataset)

