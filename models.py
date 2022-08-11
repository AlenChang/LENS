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
        self.theta = nn.Parameter(torch.randn(int(self.lens_num / 2), 1, dtype=torch.cfloat))
        self.A = torch.from_numpy(mat['A']).to(torch.cfloat).to(device)
        self.G = torch.from_numpy(mat['G']).to(torch.cfloat).to(device)
        
        self.add_weights = torch.from_numpy(mat['add_weights']).to(torch.float).to(device)
        
        # self.gain = 0.8
        # self.ns = nn.Parameter(torch.randint(1).abs())

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

        theta = torch.vstack((theta, torch.flipud(theta)))
        lens_out = torch.matmul(lens_in, torch.diag(theta.squeeze()))
        sout = torch.matmul(lens_out, torch.transpose(self.A, 0, 1))

        K = 0
        diag_out = 0
        diag_sum = 0
        
        ns = 0
        gain = 0.8
        for ti in range(-ns,ns+1):
            if(ti == 0):
                weights = -self.add_weights + self.add_weights[0][0] + 1
                weights = weights ** 2
                weights = weights / torch.max(weights)
                diag_out += gain ** np.abs(ti) * torch.sum(torch.diag(sout, ti).abs()**2 * weights)
            else:
                diag_out += gain ** np.abs(ti) * torch.sum(torch.diag(sout, ti).abs()**2)
            diag_sum += torch.sum(torch.diag(sout, ti).abs()**2)
        out = -(1+K)*diag_out \
            + K*(torch.sum(sout.abs()**2) - diag_sum)\
            + 0 * torch.var(torch.diag(sout).abs() * self.add_weights)
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
    
class Model3(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, device):
        super().__init__()
        # initialize weights with random numbers
        mat = scipy.io.loadmat('parameters.mat')

        self.array_num = mat['G'].shape[0]
        self.code_num = mat['A'].shape[0]
        self.lens_num = mat['A'].shape[1]
        self.device = device

        self.w = nn.Parameter(torch.rand(self.code_num, self.array_num, dtype=torch.cfloat))
        self.theta = nn.Parameter(torch.randn(self.lens_num, 1, dtype=torch.cfloat))
        self.A = torch.from_numpy(mat['A']).to(torch.cfloat).to(device)
        self.G = torch.from_numpy(mat['G']).to(torch.cfloat).to(device)
        self.steerVec = torch.from_numpy(mat['steerVec']).to(torch.cfloat).to(device)
        self.sweep_angle = torch.from_numpy(mat['sweep_angle']).to(device).view(-1)
        
        self.add_weights = torch.from_numpy(mat['add_weights']).to(torch.cfloat).to(device)

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

        lens_out = torch.matmul(lens_in, torch.diag(theta.squeeze()))
        # sout = torch.matmul(lens_out, torch.transpose(self.A, 0, 1))
        sout = torch.matmul(lens_out, self.steerVec)

        K = 0.1
        diag_out = 0
        diag_sum = 0
        sweep_max = torch.zeros(self.sweep_angle.shape[0]).to(self.device)
        ns = 0
        for mi in range(self.sweep_angle.shape[0]):
            angle = self.sweep_angle[mi]
            index = angle + 90
            for ti in range(-ns,ns+1):
                scale = 0.8 ** np.abs(ti)
                diag_out += scale * sout[mi, index + ti].abs() ** 2
                diag_sum += sout[mi, index + ti].abs() ** 2
                if(ti == 0):
                    sweep_max[mi] = sout[mi, index]
        out = -(1+K)*diag_out \
            + K*(torch.sum(sout.abs()**2) - diag_sum)\
            + 0 * torch.var(sweep_max * self.add_weights)
        return out
     
class Model_Basic(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, device):
        super().__init__()
        # initialize weights with random numbers
        mat = scipy.io.loadmat('parameters.mat')
        # params = scipy.io.loadmat('optimal.mat')

        self.array_num = mat['G'].shape[0]
        self.code_num = mat['A'].shape[0]
        self.lens_num = mat['steerVec'].shape[0]
        self.device = device

        
        
        # self.theta = torch.from_numpy(params['len_theta']).to(torch.cfloat).to(device)
        # self.theta = nn.Parameter(torch.from_numpy(params['len_theta']).to(torch.cfloat))
        
        # w_optimal = params['speaker_w']
        self.G = torch.from_numpy(mat['G']).to(torch.cfloat).to(device)
        self.steerVec = torch.from_numpy(mat['steerVec']).to(torch.cfloat).to(device)
        self.sweep_angle = torch.from_numpy(mat['sweep_angle']).to(device).view(-1)
        # self.add_weights = torch.from_numpy(mat['add_weights']).to(torch.cfloat).to(device)
        
        self.w = nn.Parameter(torch.rand(self.steerVec.shape[1], self.array_num, dtype=torch.cfloat))
        
        if(self.lens_num == 16):
            self.theta = nn.Parameter(torch.rand(int(self.lens_num / 2), 1, dtype=torch.cfloat))
            self.lens_type = "1D"
            self.gain = 1 / self.lens_num
        elif(self.lens_num == 256):
            self.theta = nn.Parameter(torch.rand(int(self.lens_num / 4), 1, dtype=torch.cfloat))
            self.lens_type = "2D"
            self.gain = 1 / self.lens_num
            
        # with torch.no_grad():
        #     for ti in range(self.sweep_angle.shape[0]):
        #         self.w[self.sweep_angle[ti]+90] = torch.from_numpy(w_optimal[ti]).to(torch.cfloat)
        
        # self.A = torch.from_numpy(mat['A']).to(torch.cfloat).to(device)
        
        self.add_weights = torch.from_numpy(mat['add_weights']).to(torch.float).to(device)

    def get_w(self):
        # return self.w / torch.abs(self.w)
        return self.w / torch.max(self.w.abs(), dim=1, keepdim=True)[0]
        # return self.w  / torch.sum(self.w.abs(), 1).unsqueeze(-1)
    def get_theta(self):
        if(self.lens_type == "1D"):
            return self.theta / torch.abs(self.theta)
        elif(self.lens_type == "2D"):
            theta = torch.zeros(16, 16, dtype=torch.cfloat).to(self.device)
            aa = (self.theta / self.theta.abs()).view(8, -1)
            # test_index = torch.from_numpy(np.arange(64)).view(8,8)
            # target_index = torch.zeros(16, 16)
            
            for mi in range(8):
                theta[mi, 0:8] = aa[mi]
                # target_index[mi, 0:8] = test_index[mi]
                theta[mi, 8:16] = torch.flipud(aa[mi])
                # target_index[mi, 8:16] = torch.flipud(test_index[mi])
                theta[15-mi] = theta[mi]
                # target_index[15-mi] = target_index[mi]
                
            return (theta / torch.abs(theta)).view(-1)
    
    def get_weights(self):
        x = np.arange(-90, 91)
        return -0.6 / 90 ** 2 * torch.from_numpy(x)**2 + 1
        
        


    def forward(self):
        w = self.get_w()
        theta = self.get_theta()
        if(self.lens_type == "1D"):
            theta = torch.vstack((theta, torch.flipud(theta)))
           
        # theta = self.get_theta()
        lens_in = torch.matmul(w, self.G)

        lens_out = torch.matmul(lens_in, torch.diag(theta.squeeze()))
        sout = torch.matmul(lens_out, self.steerVec)
        
        sout = sout.abs() ** 2
        sout = sout / self.array_num / self.lens_num
        
        if(self.lens_type == "1D"):
            K = 0.02
            var_gain = 100
        elif(self.lens_type == "2D"):
            K = 0.02
            var_gain = 80
        diag_out = 0
        diag_sum = 0
        weights = self.get_weights().to(self.device)
        
        diag_out += torch.sum(torch.diag(sout, 0) * weights)
        diag_sum += torch.sum(torch.diag(sout, 0))
        
        var_loss = var_gain * torch.std(torch.diag(sout) / weights)
        amp_loss = -diag_out \
            + K*(torch.sum(sout) - diag_sum)
        out = amp_loss + var_loss
        
        print("Amp loss: ", amp_loss.detach().numpy(), " Var loss: ", var_loss.detach().numpy())
        
        # out = -(1+K)*diag_out \
        #     + K*(torch.sum(sout.abs()**2) - diag_sum)\
        #     + 0 * torch.var(torch.diag(sout).abs() * self.add_weights)
        return out
    
class Model_Tune(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, device):
        super().__init__()
        # initialize weights with random numbers
        mat = scipy.io.loadmat('parameters.mat')
        params = scipy.io.loadmat('optimal.mat')

        self.array_num = mat['G'].shape[0]
        self.code_num = mat['A'].shape[0]
        self.lens_num = mat['steerVec'].shape[0]
        self.device = device
 
        # self.theta = torch.from_numpy(params['len_theta']).to(torch.cfloat).to(device)
        theta = torch.from_numpy(params['len_theta'])
        
        self.w = nn.Parameter(torch.from_numpy(params['speaker_w']).to(torch.cfloat))
        if(self.lens_num == 16):
            # self.theta = nn.Parameter(torch.randn(int(self.lens_num / 2), 1, dtype=torch.cfloat))
            self.lens_type = "1D"
            self.gain = 1
            self.theta = nn.Parameter(theta[0:int(self.lens_num / 2)].to(torch.cfloat))
        elif(self.lens_num == 256):
            # self.theta = nn.Parameter(torch.randn(int(self.lens_num / 4), 1, dtype=torch.cfloat))
            self.lens_type = "2D"
            self.gain = 1 / 500
            self.theta = nn.Parameter(theta[0][0:int(self.lens_num / 4)].to(torch.cfloat))
        
        # w_optimal = params['speaker_w']
        self.G = torch.from_numpy(mat['G']).to(torch.cfloat).to(device)  
        self.steerVec = torch.from_numpy(mat['steerVec']).to(torch.cfloat).to(device)
        self.sweep_angle = torch.from_numpy(mat['sweep_angle']).to(device).view(-1)
        # self.add_weights = torch.from_numpy(mat['add_weights']).to(torch.cfloat).to(device)
        
        # self.w = nn.Parameter(torch.rand(self.steerVec.shape[1], self.array_num, dtype=torch.cfloat))
        # self.theta = nn.Parameter(torch.randn(int(self.lens_num / 2), 1, dtype=torch.cfloat))
        # with torch.no_grad():
        #     for ti in range(self.sweep_angle.shape[0]):
        #         self.w[self.sweep_angle[ti]+90] = torch.from_numpy(w_optimal[ti]).to(torch.cfloat)
        
        # self.A = torch.from_numpy(mat['A']).to(torch.cfloat).to(device)
        
        # self.add_weights = torch.from_numpy(mat['add_weights']).to(torch.float).to(device)

    def get_w(self):
        # return self.w / torch.abs(self.w)
        return self.w / torch.max(self.w.abs(), dim=1, keepdim=True)[0]
        # return self.w  / torch.sum(self.w.abs(), 1).unsqueeze(-1)
    def get_theta(self):
        if(self.lens_type == "1D"):
            return self.theta / torch.abs(self.theta)
        elif(self.lens_type == "2D"):
            theta = torch.zeros(16, 16, dtype=torch.cfloat).to(self.device)
            aa = (self.theta / self.theta.abs()).view(8, -1)
            # test_index = torch.from_numpy(np.arange(64)).view(8,8)
            # target_index = torch.zeros(16, 16)
            
            for mi in range(8):
                theta[mi, 0:8] = aa[mi]
                # target_index[mi, 0:8] = test_index[mi]
                theta[mi, 8:16] = torch.flipud(aa[mi])
                # target_index[mi, 8:16] = torch.flipud(test_index[mi])
                theta[15-mi] = theta[mi]
                # target_index[15-mi] = target_index[mi]
                
            return (theta / torch.abs(theta)).view(-1)
    
    def get_weights(self):
        x = np.arange(-90, 91)
        return -0.2 / 90 ** 2 * torch.from_numpy(x)**2 + 1

    def forward(self):
        w = self.get_w()
        theta = self.get_theta()
        # theta = torch.vstack((theta, torch.flipud(theta)))
        if(self.lens_type == "1D"):
            theta = torch.vstack((theta, torch.flipud(theta)))
        lens_in = torch.matmul(w, self.G)
        lens_out = torch.matmul(lens_in, torch.diag(theta.squeeze()))
        sout = torch.matmul(lens_out, self.steerVec)    
        return -torch.sum(torch.diag(sout, 0).abs()**2) / self.lens_num
    
