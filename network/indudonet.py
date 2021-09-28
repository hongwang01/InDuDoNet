"""
MICCAI2021: ``InDuDoNet: An Interpretable Dual Domain Network for CT Metal Artifact Reduction''
paper link： https://arxiv.org/pdf/2109.05298.pdf
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as  F
from odl.contrib import torch as odl_torch
from .priornet import UNet
import sys
#sys.path.append("deeplesion/")
from .build_gemotry import initialization, build_gemotry
para_ini = initialization()
fp = build_gemotry(para_ini)
op_modfp = odl_torch.OperatorModule(fp)
op_modpT = odl_torch.OperatorModule(fp.adjoint)

filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9  # for initialization
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)

class InDuDoNet (nn.Module):
    def __init__(self, args):
        super(InDuDoNet, self).__init__()
        self.S = args.S                            # Stage number S includes the initialization process
        self.iter = self.S - 1                     # not include the initialization process
        self.num_u = args.num_channel + 1         # concat extra 1 term
        self.num_f = args.num_channel + 2         # concat extra 2 terms
        self.T = args.T

        # stepsize
        self.eta1const = args.eta1
        self.eta2const = args.eta2
        self.eta1 = torch.Tensor([self.eta1const])                                    # initialization for eta1  at all stages
        self.eta2 = torch.Tensor([self.eta2const])                                    # initialization for eta2 at all stages
        self.eta1S = self.make_coeff(self.S, self.eta1)                               # learnable in iterative process
        self.eta2S = self.make_coeff(self.S, self.eta2)

        # weight factor
        self.alphaconst = args.alpha
        self.alpha = torch.Tensor([self.alphaconst])
        self.alphaS = self.make_coeff(self.S, self.alpha)                             # learnable in iterative process

        # priornet
        self.priornet = UNet(n_channels=2, n_classes=1, n_filter=32)

        # proxNet for initialization
        self.proxNet_X0 = CTnet(args.num_channel + 1, self.T)                       # args.num_channel: the number of channel concatenation  1: gray CT image
        self.proxNet_S0 = Projnet(args.num_channel + 1, self.T)                     # args.num_channel: the number of channel concatenation  1: gray normalized sinogram

        # proxNet for iterative process
        self.proxNet_Xall = self.make_Xnet(self.S, args.num_channel+1, self.T)
        self.proxNet_Sall = self.make_Snet(self.S, args.num_channel+1, self.T)


        # Initialization S-domain by convoluting on XLI and SLI, respectively
        self.CX_const = filter.expand(args.num_channel, 1, -1, -1)
        self.CX = nn.Parameter(self.CX_const, requires_grad=True)
        self.CS_const = filter.expand(args.num_channel, 1, -1, -1)
        self.CS = nn.Parameter(self.CS_const, requires_grad=True)

        self.bn = nn.BatchNorm2d(1)

    def make_coeff(self, iters,const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters,-1)
        coeff = nn.Parameter(data=const_f, requires_grad = True)
        return coeff

    def make_Xnet(self, iters, channel, T):  #
        layers = []
        for i in range(iters):
            layers.append(CTnet(channel, T))
        return nn.Sequential(*layers)

    def make_Snet(self, iters, channel, T):
        layers = []
        for i in range(iters):
            layers.append(Projnet(channel, T))
        return nn.Sequential(*layers)

    def forward(self, Xma, XLI, M, Sma, SLI, Tr):
        # save mid-updating results
        ListS = []                # saving the reconstructed normalized sinogram
        ListX = []                # saving the reconstructed  CT image
        ListYS = []                # saving the reconstructed sinogram

        # with the channel concatenation and detachment operator (refer to https://github.com/hongwang01/RCDNet) for initializing dual-domain
        XZ00 = F.conv2d(XLI,  self.CX, stride=1, padding=1)
        input_Xini = torch.cat((XLI, XZ00), dim=1)             #channel concatenation
        XZ_ini = self.proxNet_X0(input_Xini)
        X0 = XZ_ini[:, :1, :, :]                              #channel detachment
        XZ = XZ_ini[:, 1:, :, :]                              #auxiliary variable in image domain
        X = X0                                                # the initialized CT image

        SZ00 = F.conv2d(SLI, self.CS, stride=1, padding=1)
        input_Sini = torch.cat((SLI, SZ00), dim=1)
        SZ_ini = self.proxNet_S0(input_Sini)
        S0 = SZ_ini[:, :1, :, :]
        SZ = SZ_ini[:, 1:, :, :]                               # auxiliary variable in sinogram domain
        S = S0                                                 # the initialized normalized sinogram
        ListS.append(S)

        # PriorNet
        prior_input = torch.cat((Xma, XLI), dim=1)
        Xs = XLI + self.priornet(prior_input)
        Y = op_modfp(F.relu(self.bn(Xs)) / 255)
        Y = Y / 4.0 * 255                                     #normalized coefficients

        # 1st iteration: Updating X0, S0-->S1
        PX= op_modfp(X/255)/ 4.0 * 255
        GS = Y * (Y*S - PX) + self.alphaS[0]*Tr * Tr * Y * (Y * S - Sma)
        S_next = S - self.eta1S[0]/10*GS
        inputS = torch.cat((S_next, SZ), dim=1)
        outS = self.proxNet_Sall[0](inputS)
        S = outS[:,:1,:,:]                                     # the updated normalized sinogram at the 1th stage
        SZ =  outS[:,1:,:,:]
        ListS.append(S)
        ListYS.append(Y*S)

        # 1st iteration: Updating X0, S1-->X1
        ESX = PX - Y*S
        GX = op_modpT((ESX/255) * 4.0)
        X_next = X - self.eta2S[0] / 10 * GX
        inputX = torch.cat((X_next, XZ), dim=1)
        outX = self.proxNet_Xall[0](inputX)
        X = outX[:, :1, :, :]                                              # the updated CT image at the 1th stage
        XZ = outX[:, 1:, :, :]
        ListX.append(X)

        for i in range(self.iter):

            # updating S
            PX = op_modfp(X / 255) / 4.0 * 255
            GS = Y * (Y * S - PX)  + self.alphaS[i+1] * Tr * Tr * Y * (Y * S - Sma)
            S_next = S - self.eta1S[i+1] / 10 * GS
            inputS = torch.cat((S_next, SZ), dim=1)
            outS = self.proxNet_Sall[i+1](inputS)
            S = outS[:, :1, :, :]
            SZ = outS[:, 1:, :, :]
            ListS.append(S)
            ListYS.append(Y * S)

            # updating X
            ESX = PX - Y * S
            GX = op_modpT((ESX / 255) * 4.0)
            X_next = X - self.eta2S[i+1] / 10 * GX
            inputX = torch.cat((X_next, XZ), dim=1)
            outX = self.proxNet_Xall[i+1](inputX)
            X = outX[:, :1, :, :]
            XZ = outX[:, 1:, :, :]
            ListX.append(X)
        return ListX, ListS, ListYS

# proxNet_S
class Projnet(nn.Module):
    def __init__(self, channel, T):
        super(Projnet, self).__init__()
        self.channels = channel
        self.T = T
        self.layer = self.make_resblock(self.T)
    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.BatchNorm2d(self.channels),
                              nn.ReLU(),
                              nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.BatchNorm2d(self.channels),
                              ))
        return nn.Sequential(*layers)

    def forward(self, input):
        S = input
        for i in range(self.T):
            S = F.relu(S + self.layer[i](S))
        return S

# proxNet_X
class CTnet(nn.Module):
    def __init__(self, channel, T):
        super(CTnet, self).__init__()
        self.channels = channel
        self.T = T
        self.layer = self.make_resblock(self.T)
    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
            ))
        return nn.Sequential(*layers)

    def forward(self, input):
        X = input
        for i in range(self.T):
            X = F.relu(X + self.layer[i](X))
        return X
