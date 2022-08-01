from email import policy
from torch import nn
import numpy as np
import torch
import scipy.io
import random
# from torch.functional import F


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
    def __init__(self, device):
        super().__init__()
        # initialize weights with random numbers
        mat = scipy.io.loadmat('parameters.mat')

        self.array_num = mat['G'].shape[0]
        self.code_num = mat['A'].shape[0]
        self.lens_num = mat['A'].shape[1]

        self.w = nn.Parameter(torch.rand(self.code_num, self.array_num, dtype=torch.cfloat))
        self.theta = nn.Parameter(torch.randn(self.lens_num, 1, dtype=torch.cfloat))
        self.A = torch.from_numpy(mat['A']).to(torch.cfloat).to(device)
        self.G = torch.from_numpy(mat['G']).to(torch.cfloat).to(device)
        
        self.add_weights = torch.from_numpy(mat['add_weights']).to(torch.cfloat).to(device)
        # self.G = torch.rand(self.array_num, self.lens_num, dtype=torch.cfloat)
    # def reset(self, device):
    #     self.w = nn.Parameter(torch.rand(self.code_num, self.array_num, dtype=torch.cfloat)).to(device)
    #     self.theta = nn.Parameter(torch.randn(self.lens_num, 1, dtype=torch.cfloat)).to(device)
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
        # entropy = 0.0
        # for input_tensor in sout:
        #     entropy += calc_entropy(input_tensor.abs())

        # weight_entropy = calc_entropy(w.abs().squeeze())
        # out = -torch.sum(torch.diag(sout).abs())
        # print("diag: ", torch.diag(sout).abs())
        K = 0.1
        diag_out = 0
        diag_sum = 0
        ns = 1
        for ti in range(-ns,ns+1):
            diag_out += 0.8 ** np.abs(ti) \
                * torch.sum(torch.diag(sout, ti).abs()**2)
            diag_sum += torch.sum(torch.diag(sout, ti).abs()**2)
        out = -(1+K)*diag_out \
            + K*(torch.sum(sout.abs()**2) - diag_sum)\
            + 1000 * torch.var(torch.diag(sout).abs() * self.add_weights)
        return out
    
    
class Model2(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, device):
        super().__init__()
        # initialize weights with random numbers
        mat = scipy.io.loadmat('parameters.mat')

        self.array_num = mat['G'].shape[0]
        self.code_num = mat['A'].shape[0]
        self.lens_num = mat['A'].shape[1]

        self.A = torch.from_numpy(mat['A']).to(torch.cfloat).to(device)
        self.G = torch.from_numpy(mat['G']).to(torch.cfloat).to(device)

        self.w = torch.rand(self.code_num, self.array_num, dtype=torch.cfloat).to(device)
        self.theta = torch.randn(self.lens_num, 1, dtype=torch.cfloat).to(device)
    

        self.hidden = 128
        self.net_beampattern = nn.Sequential(
            nn.Linear(self.code_num**2, self.hidden),
            nn.ReLU()
        )
        
        self.net_w = nn.Sequential(
            nn.Linear(self.code_num*self.array_num*2, self.hidden),
            nn.ReLU()
        )
        
        self.net_theta = nn.Sequential(
            nn.Linear(self.lens_num * 2, 16),
            nn.ReLU()
        )
        
        self.policy = nn.Sequential(
            nn.Linear(self.hidden+self.hidden+16, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU()
        )
        self.policy_w = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.code_num*self.array_num*2)
        )
        
        self.policy_theta = nn.Sequential(
            # nn.Linear(self.hidden, self.hidden),
            # nn.ReLU(),
            nn.Linear(self.hidden, self.lens_num*2)
        )
    def reset(self, device):
        self.w = torch.rand(self.code_num, self.array_num, dtype=torch.cfloat).to(device)
        self.theta = torch.randn(self.lens_num, 1, dtype=torch.cfloat).to(device)

    def get_w(self, w=None):
        if(w==None):
            w = self.w
        return w / torch.abs(w)
        # return w / torch.max(w.abs(), dim=1, keepdim=True)[0]
        # return w  / torch.sum(w.abs(), 1).unsqueeze(-1)
    def get_theta(self, theta=None):
        if(theta==None):
            theta = self.theta
        return self.theta / torch.abs(self.theta)
    
    def compute_out_mat(self, w, theta):
        # w = self.get_w()
        # theta = self.get_theta()
        lens_in = torch.matmul(w, self.G)
        lens_out = torch.matmul(lens_in, torch.diag(theta.squeeze()))
        sout = torch.matmul(lens_out, torch.transpose(self.A, 0, 1))
        return sout
    def compute_award(self, sout, K):
        return -(1+K)*torch.sum(torch.diag(sout).abs()) + K*torch.sum(sout.abs())
    def get_feature_w(self):
        w = self.get_w().view(-1)
        feature_w = torch.hstack((torch.real(w), torch.imag(w)))
        return feature_w
    def get_feature_theta(self):
        theta = self.get_theta().view(-1)
        feature_theta = torch.hstack((torch.real(theta), torch.imag(theta)))
        return feature_theta
    def get_feature_bp(self, bp):
        return bp.abs().view(-1)
        



    def forward(self):
        sout1 = self.compute_out_mat(self.w, self.theta)
        sout1 = torch.rand(self.code_num, self.code_num).to('cuda')
        feature_w = nn.functional.normalize(self.net_w(self.get_feature_w()), dim=0)
        feature_theta = nn.functional.normalize(self.net_theta(self.get_feature_theta()), dim=0)
        feature_bp = nn.functional.normalize(self.net_beampattern(self.get_feature_bp(sout1)), dim=0)
        
        features = torch.hstack((feature_w, feature_theta, feature_bp))
        
        features = self.policy(features)
        dw = self.policy_w(features)
        dw = dw.view(2, -1)
        dw = dw[0] + dw[1]*1j
        
        w = dw.view(self.w.shape) + self.w
        dtheta = self.policy_theta(features)
        dtheta = dtheta.view(2, -1)
        dtheta = dtheta[0] + dtheta[1]*1j
        
        theta = dtheta.view(self.theta.shape) + self.theta
        w = self.get_w(w)
        theta = self.get_theta(theta)
        
        sout = self.compute_out_mat(w, theta)
        K = 0.05
        out = self.compute_award(sout, K)
        self.w = w.detach()
        self.theta = theta.detach()
        return out
