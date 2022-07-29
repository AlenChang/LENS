# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from copy import copy
import seaborn as sns
import scipy.io
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

sns.set_style("whitegrid")

mat = scipy.io.loadmat('parameters.mat')
A = torch.from_numpy(mat['A']).to(torch.cfloat)

# %%

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
    def __init__(self, array_num, code_num, lens_num):
        super().__init__()
        # initialize weights with random numbers
        self.array_num = array_num
        self.code_num = code_num
        self.lens_num = lens_num

        self.w = nn.Parameter(torch.rand(self.code_num, self.array_num, dtype=torch.cfloat))

        self.theta = nn.Parameter(torch.randn(self.lens_num, 1, dtype=torch.cfloat))

        # mat = scipy.io.loadmat('parameters.mat')
        self.A = torch.from_numpy(mat['A']).to(torch.cfloat)
        # self.A = torch.rand(self.code_num, self.lens_num, dtype=torch.cfloat)

        self.G = torch.from_numpy(mat['G']).to(torch.cfloat)
        # self.G = torch.rand(self.array_num, self.lens_num, dtype=torch.cfloat)
    def get_w(self):
        # return self.w / torch.abs(self.w)
        return self.w / torch.max(self.w.abs(), dim=1, keepdim=True)[0]
        # return self.w  / torch.sum(self.w.abs(), 1).unsqueeze(-1)
    def get_theta(self):
        return self.theta / torch.abs(self.theta)


    def forward(self):
        w = self.get_w()
        theta = self.get_theta()
        lens_in = torch.matmul(w, self.G)
        # lens_in = lens_in / torch.sum(torch.abs(lens_in), 1).unsqueeze(-1)

        lens_out = torch.matmul(lens_in, torch.diag(theta.squeeze()))
        sout = torch.matmul(lens_out, torch.transpose(self.A, 0, 1))

        # sout = torch.matmul(sin, theta)
        entropy = 0.0
        for input_tensor in sout:
            entropy += calc_entropy(input_tensor.abs())

        # weight_entropy = calc_entropy(w.abs().squeeze())
        out = -torch.diag(sout).norm(dim=0, p=1)/self.code_num + 0.1 * entropy
        # out = -torch.diag(sout).norm(dim=0, p=1)/self.code_num + 0.1*(torch.sum(sout.abs(), dim=1) - torch.diag(sout.abs())).norm(dim=0, p=1)/self.code_num
        return out

    


def training_loop(model, optimizer, n=2000):
    "Training loop for torch model."
    losses = []
    for i in range(n):
        loss = model()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)
        if(i % 200 == 0):
            print("iter: ", i,". loss: ", loss.detach().numpy())
    return losses

num_direcgtion = 17
num_speaker = 9
num_cell = 16
m = Model(num_speaker, num_direcgtion, num_cell)
# %%
# instantiate model

# Instantiate optimizer

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
optimizer = torch.optim.Adam(m.parameters(), lr=0.1)

# n = 1000
# iters = 10
# for ni in range(iters):
#     if(ni % 2 ==0):
#         m.w.requires_grad = True
#         m.theta.requires_grad = False
#     else:
#         m.w.requires_grad = False
#         m.theta.requires_grad = True
#     losses = training_loop(m, optimizer, n)

    
losses = training_loop(m, optimizer, 8000)



# losses = []
# for i in range(n):
#     loss = m()
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     losses.append(loss.abs())
#     # scheduler.step()
#     if(i % 200 == 0):
#         print("loss: ", loss.abs().detach().numpy())
    
plt.figure(figsize=(14, 7))
plt.plot(losses)
# print(m.weights)

# %%
# print estimated weigths
w = m.get_w().detach().numpy()
theta = m.get_theta().detach().numpy()

# print("Weights: ", w)
mdic = {"speaker_w": w, "len_theta":theta}
scipy.io.savemat("optimal.mat", mdic)
# %%
