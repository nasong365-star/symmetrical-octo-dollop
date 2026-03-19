import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn
import sys
import numpy.linalg as lin

class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=64):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list):
        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb


class TrustworthyNet_tensor(nn.Module):
    def __init__(self, nfeats, n_view, n_classes, n, PHI, args, device):
        """
        build TrustworthyNet for classfier.
        :param nfeats: list of feature dimensions for each view
        :param n_view: number of views
        :param n_classes: number of clusters
        :param n: number of samples
        :param args: Relevant parameters required to build the network
        """
        super(TrustworthyNet_tensor, self).__init__()
        self.n_classes = n_classes
        #  number of differentiable blocks
        self.block = args.block
        # the initial value of the threshold
        self.theta = nn.Parameter(torch.FloatTensor([args.thre]), requires_grad=True).to(device)
        self.theta_n = nn.Parameter(torch.FloatTensor([args.thre_n]), requires_grad=True).to(device)
        self.n_view = n_view
        self.n = n
        self.phi = PHI
        self.HH_init = []
        self.alpha = args.alpha
        self.delta = args.delta
        self.beta = args.beta
        self.fusion_type = args.fusion_type
        self.input_type = args.input_type
        self.bn_input_01 = nn.BatchNorm1d(self.n_classes, momentum=0.5).to(device)
        self.device = device
        self.fusionlayer = FusionLayer(n_view, self.fusion_type, n_classes, hidden_size=64)
        nv = 0

        self.blocks = nn.ModuleList([Block( n_classes, feat, device) for feat in nfeats]).to(device)
        for ij in range(n_view):
            nv += nfeats[ij]
            # print(nn.Linear(nfeats[ij], n_classes).to(device))
            self.HH_init.append(nn.Linear(nfeats[ij], n_classes).to(device))
        self.cc = nn.Linear(nv, self.n_classes).to(device)

    def self_active_tsvd(self, H_list):
        # 得到Hv H_list - > [h1....hv]
        X = torch.zeros((self.n_classes, self.n_view, self.n), requires_grad=True)
        Xf = torch.zeros((self.n_classes, self.n_view, self.n), requires_grad=True)
        X = torch.complex(X, Xf)

        # 扩展到三维张量H‘ n x c x v rotate -> c x v x n

        H_1 = torch.stack(H_list, dim=-1)
        H_rotate = H_1.transpose(dim0=0, dim1=1)
        H_rotate = H_rotate.transpose(dim0=1, dim1=2)
        H_fft = torch.fft.fft((H_rotate))

        def usv(H_slice,slice):
            U, S, Vt = torch.linalg.svd(H_slice)
            # r = length(find(S>rho)); matlab中S转为了一列，python原始便是一行
            # mm = torch.where(S > int(self.theta_n))
            S = F.leaky_relu(S - self.theta_n)
            mm = torch.where(S > 0)
            r = mm[0].shape[0]

            if (r >= 1):
                # S = S[:r] - int(self.theta_n)
                # S = F.leaky_relu(S[:r] - self.theta_n)
                # S_ = torch.diag_embed(S)
                S_ = torch.diag_embed(S[:r])
                Sf = torch.zeros_like(S_, requires_grad=True)
                S_complex = torch.complex(S_, Sf)
                # assert U[:, 0:r].shape[1] == S_complex.shape[0], "U and S_complex dimensions do not match"
                # assert Vt[0:r, :].shape[0] == S_complex.shape[0], "Vt and S_complex dimensions do not match"
                X[:, :, slice] = torch.mm(torch.mm(U[:, 0:r], S_complex), Vt[0:r, :])
            return X[:, :, slice]
        # first slice
        X[:, :, 0] = usv(H_fft[:, :, 0],0)

        if (self.n % 2 == 0):
            halfn = int(self.n / 2)
        else:
            halfn = int(self.n / 2 + 1)

        for i in range(1, halfn):
            X[:, :, i] = usv(H_fft[:, :, i], i)
            X[:, :, self.n - i] = torch.conj(X[:, :, i])

        if (self.n % 2 == 0):
            i = int(self.n / 2)
            X[:, :, i] = usv(H_fft[:, :, i], i)

        X = torch.fft.ifft(X, dim=2)
        X_n = torch.real(X)
        H_list1 = []
        X_final = X_n.transpose(dim0=0, dim1=1)
        X_final = X_final.transpose(dim0=1, dim1=2)
        for i in range(self.n_view):
            H_list1.append(torch.Tensor(X_final[i, :, :]).to(self.device))
        return H_list1

    def self_active_l1(self, u):
        return F.selu(u - self.theta) - F.selu(-1.0 * u - self.theta)

    def forward(self, features, sim, lap, lap2):
        output_h = []
        result_h = []
        h = dict()
        XV = torch.cat(features, dim=1)
        XV = self.cc(XV)

        for j in range(0, self.n_view):
            out_tmp = self.HH_init[j](features[j] / 1.0)
            output_h.append(out_tmp.to(self.device))    # out_h --> [h1-hv]
        result_h.append(self.fusionlayer(output_h).to(self.device))     # H0
        for i in range(0, self.block):
            H_list = []
            for j in range(0, self.n_view):
                h[j] = self.blocks[j](self.alpha, self.delta, self.beta, output_h[j], lap[j], features[j] / 1.0, self.phi, lap2[j])
                # 每一个视角都经过l1激活函数，最后融合。（BMRL是融合后经过l1函数）
                h[j] = self.self_active_l1(h[j])
                H_list.append(h[j])

            H_list = self.self_active_tsvd(H_list)
            result_h.append(self.bn_input_01(self.fusionlayer(H_list)))      # h1

        # additional step, 看性能的情况决定是否去除
        result_h[-1] += XV

        return result_h[1:]

class Block(Module):
    # differentiable network block
    def __init__(self, out_features, nfea, device):
        super(Block, self).__init__()
        self.W_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.W = nn.Linear(out_features, out_features).to(device)

        self.U_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.U = nn.Linear(out_features, out_features).to(device)

        self.G_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.G = nn.Linear(out_features, out_features).to(device)

        self.O_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.O = nn.Linear(out_features, out_features).to(device)

        self.S_norm = nn.BatchNorm1d(nfea, momentum=0.6).to(device)
        self.S = nn.Linear(nfea, out_features).to(device)

        self.device = device

    def forward(self, alpha, delta, beta, input, lap, fea, PHI, lap2):
        input1 = self.W(self.W_norm(input))  # input = h
        input2 = self.S(self.S_norm(fea))

        lap_tmp = fea.mm(lap.mm(fea.t()))
        output = torch.mm(lap_tmp, input)
        output = self.G(self.G_norm(output))
        # sam = 0
        o = torch.mm(lap2, input)
        sam = self.U(self.U_norm(o))
        PHI = torch.FloatTensor(PHI).to(self.device)
        ortho = self.O(self.O_norm(torch.mm(PHI, input)))
        output = input1 + input2 - output * alpha - ortho * delta - sam * beta
        return output
