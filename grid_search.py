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
# setup_seed(20)

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
            best_model = model
            max_loss = loss
        if(i % 1000 == 0):
            print(COMMENTS + " iter: ", i,". loss: ", loss.cpu().detach().numpy())
            # print(COMMENTS + " iter: ", i,".")
    torch.save(best_model.state_dict(), 'best_model.pkl')
    return losses, max_loss



# %%
# print estimated weigths
# w = m.get_w().cpu().detach().numpy()
# theta = m.get_theta().cpu().detach().numpy()

# # print("Weights: ", w)
# mdic = {"speaker_w": w, "len_theta":theta}
# scipy.io.savemat("optimal.mat", mdic)
# %%
with open("record.txt", "w") as file:
    file.write("Loss\n")
    
loss_hist = []
counter = 0
for side_lobe_gain in np.arange(0.025, 0.046, 0.0001):
    print("Processing test ", side_lobe_gain)
    setup_seed(20)

    # filename = "vary_speaker_num/parameters"+str(ti)+".mat"
    filename = "parameters.mat"
    model = Model_Side_lobe_gain(filename, device).to(device)
    model.set_side_lobe_gain(side_lobe_gain)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
        step_size=8000, gamma=0.2)
    max_loss = 9999
    # model.w.requires_grad = True
    # model.theta.requires_grad = False
    losses, max_loss = training_loop(model,\
            optimizer, scheduler, 30001, max_loss)


    model.load_state_dict(torch.load('best_model.pkl'))
    # print estimated weigths
    w = model.get_w().cpu().detach().numpy()
    theta = model.get_theta().cpu().detach().numpy()

    # print("Weights: ", w)
    mdic = {"speaker_w": w, "len_theta":theta}
    scipy.io.savemat("optimal.mat", mdic)

    # %% Tuning
    max_loss = 9999
    model_tune = Model_Tune(filename, device).to(device)
    model_tune.load_state_dict(torch.load('best_model.pkl'))
    optimizer = torch.optim.Adam(model_tune.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
        step_size=5000, gamma=0.1)
    model_tune.w.requires_grad = True
    model_tune.theta.requires_grad = False

    losses, max_loss = training_loop(model_tune,\
            optimizer, scheduler, 5001, max_loss)

    loss_hist.append(max_loss.detach().numpy())
    with open("record.txt", "a") as file:
        file.write(str(side_lobe_gain)+" " + str(max_loss.detach().numpy())+"\n")
        print("Save record success")

    model_tune.load_state_dict(torch.load('best_model.pkl'))
    w = model_tune.get_w().cpu().detach().numpy()
    theta = model_tune.get_theta().cpu().detach().numpy()

    # print("Weights: ", w)
    mdic = {"speaker_w": w, "len_theta":theta}
    scipy.io.savemat("mat/optimal_"+ str(counter) + ".mat", mdic)
    counter += 1
    # scipy.io.savemat("vary_speaker_num/optimal"+str(ti)+".mat", mdic)
    # scipy.io.savemat("optimal.mat", mdic)

    # lossdic = {"loss", np.array(loss_hist)}
    # scipy.io.savemat("record_loss.mat", lossdic)
