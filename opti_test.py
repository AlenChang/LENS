# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from copy import copy
import seaborn as sns
sns.set_style("whitegrid")
n = 1000
noise = torch.Tensor(np.random.normal(0, 0.02, size=n))
x = torch.arange(n)
a, k, b = 0.7, .01, 0.2
y = a * np.exp(-k * x) + b + noise
plt.figure(figsize=(14, 7))
plt.scatter(x, y, alpha=0.4)

# %%
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

        self.theta = nn.Parameter(torch.rand(self.lens_num, 1, dtype=torch.cfloat))

        self.A = torch.rand(self.code_num, self.lens_num, dtype=torch.cfloat)

        self.G = torch.rand(self.array_num, self.lens_num, dtype=torch.cfloat)

    def forward(self):
        w = self.w  / torch.sum(self.w.abs(), 1).unsqueeze(-1)
        theta = self.theta / torch.abs(self.theta)
        sin = torch.matmul(w, self.G) * self.A

        sout = torch.matmul(sin, theta)
        return -sout.norm(dim=0, p=1)/self.code_num

def training_loop(model, optimizer, n=5000):
    "Training loop for torch model."
    losses = []
    for i in range(n):
        loss = model()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.abs())
        print("loss: ", loss.abs().detach().numpy())
    return losses

m = Model(9, 17, 16)
# %%
# instantiate model

# Instantiate optimizer
opt = torch.optim.Adam(m.parameters(), lr=0.002)
losses = training_loop(m, opt)
plt.figure(figsize=(14, 7))
plt.plot(losses)
# print(m.weights)

# %%
# preds = m(x)
# plt.figure(figsize=(14, 7))
# plt.scatter(x, preds.detach().numpy())
# plt.scatter(x, y, alpha=.3)