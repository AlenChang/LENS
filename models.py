from email import policy
from torch import nn
import numpy as np
import torch
import scipy.io
import random


def calc_entropy(input_tensor):
    lsm = nn.LogSoftmax()
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean()
    return entropy

class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self):
        super().__init__()
        # initialize weights with random numbers
        mat = scipy.io.loadmat('parameters.mat')

        self.array_num = mat['G'].shape[0]
        self.code_num = mat['A'].shape[0]
        self.lens_num = mat['A'].shape[1]

        self.w = nn.Parameter(torch.rand(self.code_num, self.array_num, dtype=torch.cfloat))

        self.theta = nn.Parameter(torch.randn(self.lens_num, 1, dtype=torch.cfloat))

        # mat = scipy.io.loadmat('parameters.mat')
        self.A = torch.from_numpy(mat['A']).to(torch.cfloat)
        # self.A = torch.rand(self.code_num, self.lens_num, dtype=torch.cfloat)

        self.G = torch.from_numpy(mat['G']).to(torch.cfloat)
        # self.G = torch.rand(self.array_num, self.lens_num, dtype=torch.cfloat)
    def get_w(self):
        return self.w / torch.abs(self.w)
        # return self.w / torch.max(self.w.abs(), dim=1, keepdim=True)[0]
        # return self.w  / torch.sum(self.w.abs(), 1).unsqueeze(-1)
    def get_theta(self):
        return self.theta / torch.abs(self.theta)


    def forward(self):
        w = self.get_w()
        theta = self.get_theta()
        lens_in = torch.matmul(w, self.G)
        # print("lens_in: ", lens_in.abs())
        # lens_in = lens_in / torch.sum(torch.abs(lens_in), 1).unsqueeze(-1)

        lens_out = torch.matmul(lens_in, torch.diag(theta.squeeze()))
        sout = torch.matmul(lens_out, torch.transpose(self.A, 0, 1))

        # sout = torch.matmul(sin, theta)
        entropy = 0.0
        for input_tensor in sout:
            entropy += calc_entropy(input_tensor.abs())

        # weight_entropy = calc_entropy(w.abs().squeeze())
        # out = -torch.sum(torch.diag(sout).abs())
        # print("diag: ", torch.diag(sout).abs())
        K = 0.5
        out = -(1+K)*torch.sum(torch.diag(sout).abs()) + K*torch.sum(sout.abs())
        return out
    
    
class Model2(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self):
        super().__init__()
        # initialize weights with random numbers
        mat = scipy.io.loadmat('parameters.mat')

        self.array_num = mat['G'].shape[0]
        self.code_num = mat['A'].shape[0]
        self.lens_num = mat['A'].shape[1]

        self.A = torch.from_numpy(mat['A']).to(torch.cfloat)
        self.G = torch.from_numpy(mat['G']).to(torch.cfloat)

        self.w = torch.rand(self.code_num, self.array_num, dtype=torch.cfloat)
        self.theta = torch.randn(self.lens_num, 1, dtype=torch.cfloat)

        linear_in = self.code_num**2 + self.code_num * self.array_num + self.lens_num
        policy = nn.Sequential(
            nn.Linear(linear_in, linear_in / 2)
        )


    def get_w(self):
        return self.w / torch.abs(self.w)
        # return self.w / torch.max(self.w.abs(), dim=1, keepdim=True)[0]
        # return self.w  / torch.sum(self.w.abs(), 1).unsqueeze(-1)
    def get_theta(self):
        return self.theta / torch.abs(self.theta)
    
    def compute_out_mat(self):
        w = self.get_w()
        theta = self.get_theta()
        lens_in = torch.matmul(w, self.G)
        lens_out = torch.matmul(lens_in, torch.diag(theta.squeeze()))
        sout = torch.matmul(lens_out, torch.transpose(self.A, 0, 1))
        return sout
    def compute_award(self, sout, K):
        return -(1+K)*torch.sum(torch.diag(sout).abs()) + K*torch.sum(sout.abs())


    def forward(self):
        sout = self.compute_out_mat()

        K = 0.5
        out = self.compute_award(sout, K)
        return out
