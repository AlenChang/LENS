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
from models import *

if(torch.cuda.is_available()):
    device = 'cuda'
else:
    device = 'cpu'
device = 'cpu'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

sns.set_style("whitegrid")


# %%


    


def training_loop(model, optimizer,\
    scheduler = None, n=2000, max_loss=999, COMMENTS = " "):
    "Training loop for torch model."
    losses = []
    
    for i in range(n):
        loss = model()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
        if(scheduler is not None):
            scheduler.step()
        if(loss < max_loss):
            torch.save(model.state_dict(), 'best_model.pkl')
            max_loss = loss
        if(i % 100 == 0):
            print(COMMENTS + " iter: ", i,". loss: ", loss.cpu().detach().numpy())
            # print(COMMENTS + " iter: ", i,".")
    return losses, max_loss

m = Model(device).to(device)
# %%
# instantiate model

# Instantiate optimizer

# optimizer = torch.optim.Adam(m.parameters(), lr=0.005)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
#     step_size=4000, gamma=0.2)

# n = 2000
# iters = 21

# for ni in range(iters):
#     m.train()
#     if(ni % 2 ==0):
#         m.w.requires_grad = True
#         m.theta.requires_grad = False
#         COMMENTS = "w"
#     else:
#         m.w.requires_grad = False
#         m.theta.requires_grad = True
#         COMMENTS = "theta"
#     losses, max_loss = training_loop(m, optimizer, n, max_loss, COMMENTS)
#     m.load_state_dict(torch.load('best_model.pkl'))

# m.theta.requires_grad = False

# repeat_num = 1
# for ti in range(repeat_num):
#     print("test ", ti)
#     # m.reset(device)
#     max_loss = 9999
#     losses, max_loss = training_loop(m,\
#         optimizer, scheduler, 4501, max_loss)
#     m.load_state_dict(torch.load('best_model.pkl'))



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
    
# plt.figure(figsize=(14, 7))
# plt.plot(losses)
# plt.savefig("./figs/loss.png") 
# plt.show()
# print(m.weights)

# %%
# print estimated weigths
# w = m.get_w().cpu().detach().numpy()
# theta = m.get_theta().cpu().detach().numpy()

# # print("Weights: ", w)
# mdic = {"speaker_w": w, "len_theta":theta}
# scipy.io.savemat("optimal.mat", mdic)
# %%
# print('Tuning w with given theta')
model = Model_Basic(device).to(device)
# instantiate model

# Instantiate optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
    step_size=4000, gamma=0.2)

max_loss = 9999

# model.w.requires_grad = True
# model.theta.requires_grad = False
losses, max_loss = training_loop(model,\
        optimizer, scheduler, 20001, max_loss)


# print estimated weigths
w = model.get_w().cpu().detach().numpy()
theta = model.get_theta().cpu().detach().numpy()

# print("Weights: ", w)
mdic = {"speaker_w": w, "len_theta":theta}
scipy.io.savemat("optimal.mat", mdic)

# # %% Tuning
# max_loss = 9999
# model_tune = Model_Tune(device).to(device)
# optimizer = torch.optim.Adam(model_tune.parameters(), lr=0.0001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
#     step_size=5000, gamma=0.1)
# model_tune.w.requires_grad = True
# model_tune.theta.requires_grad = False
# losses, max_loss = training_loop(model_tune,\
#         optimizer, scheduler, 8001, max_loss)
# # for ni in range(3):
# #     model.train()
# #     if(ni % 2 ==0):
# #         model.w.requires_grad = True
# #         model.theta.requires_grad = False
# #         COMMENTS = "w"
# #     else:
# #         model.w.requires_grad = False
# #         model.theta.requires_grad = True
# #         COMMENTS = "theta"
# #     losses, max_loss = training_loop(model,\
# #             optimizer, scheduler, 8001, max_loss)
# #     model.load_state_dict(torch.load('best_model.pkl'))
# model_tune.load_state_dict(torch.load('best_model.pkl'))
# w = model_tune.get_w().cpu().detach().numpy()
# theta = model_tune.get_theta().cpu().detach().numpy()

# # print("Weights: ", w)
# mdic = {"speaker_w": w, "len_theta":theta}
# scipy.io.savemat("optimal.mat", mdic)
