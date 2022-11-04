from email import policy
from xml.etree.ElementPath import find
from torch import nn
import numpy as np
import torch
import scipy.io
import random
from scipy.signal import find_peaks
# from torch.functional import F

class Model_Basic(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, filename, device):
        super().__init__()
        # initialize weights with random numbers
        mat = scipy.io.loadmat(filename)
        # params = scipy.io.loadmat('optimal.mat')

        self.array_num = mat['G'].shape[0]
        self.code_num = mat['A'].shape[0]
        self.lens_num = mat['steerVec'].shape[0]
        self.device = device
        
        
        self.use_lagrange = False
        self.need_var_gain = True
        
        self.speaker_locs = torch.from_numpy(mat['speaker']['locs'][0][0]).to(device)
        self.lens_locs = torch.from_numpy(mat['target']['locs'][0][0]).to(device)
        self.speaker_locs = self.speaker_locs.unsqueeze(dim=1)
        self.lens_locs = self.lens_locs.unsqueeze(dim=0)
        self.G = torch.ones(self.array_num, self.lens_num).to(torch.cfloat)

        # self.G = torch.from_numpy(mat['G']).to(torch.cfloat).to(device)
        self.fc = torch.from_numpy(mat['speaker']['fc'][0][0][0].astype(np.float32))
        self.c = torch.from_numpy(mat['speaker']['c'][0][0][0].astype(np.float32))
        self.lambda1 = self.c / self.fc
        self.aparture = (self.array_num - 1) * self.lambda1
        
        
        
        
        self.steerVec = torch.from_numpy(mat['steerVec']).to(torch.cfloat).to(device)
        self.sweep_angle = torch.from_numpy(mat['sweep_angle']).to(device).view(-1)
        # self.add_weights = torch.from_numpy(mat['add_weights']).to(torch.cfloat).to(device)
        self.locs = nn.Parameter(self.speaker_locs[:,:,1].to(device))
        
        self.w = nn.Parameter(torch.rand(int((self.steerVec.shape[1]+1) / 2), self.array_num, dtype=torch.cfloat))
        
        mask = scipy.io.loadmat('mask.mat')
        self.mask = torch.from_numpy(mask['mask']).to(device)
        
        self.lens_type = mat['lens_dimension']
        if(self.lens_type == "1D"):
            self.theta = nn.Parameter(torch.rand(int(self.lens_num / 2), 1, dtype=torch.cfloat))
            
            self.gain = 1 / self.lens_num
        elif(self.lens_type == "2D"):
            self.theta = nn.Parameter(torch.rand(int(self.lens_num / 4), 1, dtype=torch.cfloat))
            self.lens_width = int(np.sqrt(self.lens_num))
            self.gain = 1 / self.lens_num
        else:
            print("Wrong lens dimension.")
            return
            
        self.add_weights = torch.from_numpy(mat['add_weights']).to(torch.float).to(device)
        
        

    def get_w(self):
        # return self.w / torch.abs(self.w)
        # return 
        # w = self.w / torch.abs(self.w)
        # w = self.w / torch.max(self.w.abs(), dim=1, keepdim=True)[0]
        
        w_abs = self.w.abs()
        w_abs = torch.clamp(w_abs, min= 1)
        w = self.w / w_abs
        # w = self.w / torch.max(self.w.abs(), dim=1, keepdim=True)[0]
        w_half = torch.flipud(torch.fliplr(w))
        return torch.vstack((w, w_half[1:]))

        # return self.w  / torch.sum(self.w.abs(), 1).unsqueeze(-1)
    def get_theta(self):
        if(self.lens_type == "1D"):
            theta = torch.vstack((self.theta, torch.flipud(self.theta)))
            with torch.no_grad():
                return theta / torch.abs(theta)
                # return theta
        elif(self.lens_type == "2D"):
            theta = torch.zeros(self.lens_width, self.lens_width, dtype=torch.cfloat).to(self.device)
            
            sub_mat_width = int(self.lens_width / 2)
            
            if(not self.use_lagrange):
                theta_squeeze = (self.theta / self.theta.abs()).view(sub_mat_width, -1)
            else:
                theta_squeeze = self.theta.view(sub_mat_width, -1)
            # test_index = torch.from_numpy(np.arange(64)).view(8,8)
            # target_index = torch.zeros(16, 16)
            
            for mi in range(sub_mat_width):
                theta[mi, 0:sub_mat_width] = theta_squeeze[mi]
                # target_index[mi, 0:8] = test_index[mi]
                theta[mi, sub_mat_width:self.lens_width] = torch.flipud(theta_squeeze[mi])
                # target_index[mi, 8:16] = torch.flipud(test_index[mi])
                theta[self.lens_width-1-mi] = theta[mi]
                # target_index[15-mi] = target_index[mi]
                
            return theta.view(-1)
    
    def get_weights(self):
        max_angle = 60
        atten = -0.3
        x = np.arange(-max_angle, max_angle+1)
        return atten / max_angle ** 2 * torch.from_numpy(x)**2 + 1

    def update_G(self):
        # self.update_speaker_locs()
        # print(self.locs)
        self.speaker_locs[:,:,1] = self.locs
        d = torch.norm(self.speaker_locs - self.lens_locs, dim=2).to(torch.float32)
        self.G = torch.exp(-1j*2*torch.pi*self.fc/self.c*d) / (2*torch.pi*d)
        return
    
    # def update_speaker_locs(self):
    #     spacing = self.normalize_spacing()
    #     self.speaker_locs[0,:,1] = self.aparture / 2
    #     self.speaker_locs[1:,:,1] = self.aparture / 2 - spacing
        
 
    
    # def normalize_spacing(self):
    #     spacing = self.spacing / (2*self.spacing[0]+2*self.spacing[1]+self.spacing[2]) * self.aparture
    #     print(2*spacing[0] + 2*spacing[1] + spacing[2])
    #     return torch.cumsum(torch.vstack((spacing, torch.fliplr(spacing[0:2]))), dim=0)

    def forward(self):
        w = self.get_w()
        theta = self.get_theta()
        # theta = self.get_theta()
        self.update_G()
        lens_in = torch.matmul(w, self.G)

        lens_out = torch.matmul(lens_in, torch.diag(theta.squeeze()))
        sout = torch.matmul(lens_out, self.steerVec)
        
        sout = sout.abs() ** 2
        sout = sout / self.array_num / self.lens_num
        
        if(self.lens_type == "1D"):
            K = 0.01
            var_gain = 50
        elif(self.lens_type == "2D"):
            K = 0.04
            var_gain = 100
        diag_out = 0
        diag_sum = 0
        weights = self.get_weights().to(self.device)
        
        diag_out += torch.sum(torch.diag(sout, 0))
        
        if(self.need_var_gain):
            var_loss = var_gain * torch.std(torch.diag(sout) / weights)
            
        
        use_mini_max = False
        use_sidelob_cancel = True
        use_adaptive_minimax = False
        use_minimal_mean = False
        use_mini_max_second_peaks = False
        
        if(not use_mini_max & (not use_sidelob_cancel) & (not use_adaptive_minimax) & (not use_minimal_mean) & (not use_mini_max_second_peaks)):
            # second_pks = 0
            # # print(self.code_num)
            # for i in range(self.code_num):
            #     pks = torch.argmax(sout[i])
            #     if(pks != i):
            #         second_pks += sout[i][pks]
            amp_loss = -diag_out
            
        if(use_minimal_mean):
            out = 0
            for i in range(self.code_num):
                # pks = torch.sum(sout[i])
                # tmp = sout[i] / pks
                out = out + torch.log(sout[i][i] / torch.mean(sout[i])) + torch.log(sout[i][i])
            amp_loss = -0.02*out
                
                
        if(use_adaptive_minimax):
            side_lobe = sout * self.mask
            amp_loss = -diag_out \
            + 0.05*side_lobe.sum()
            # print(amp_loss)
        
        if(use_sidelob_cancel):
            ignore_angle = 10
            for ti in range(-ignore_angle,ignore_angle+1):
                diag_sum += torch.sum(torch.diag(sout, ti))
                
            amp_loss = -diag_out \
            + 0.07*(torch.sum(sout) - diag_sum)
        
        if(use_mini_max):
            second_pks = 0
            # print(self.code_num)
            for i in range(self.code_num):
                pks, _ = find_peaks(sout[i].detach().numpy())
                values, index = sout[i][pks].abs().sort()
                if(index.shape[0]>=1):
                    if(pks[index[0]] == i):
                        if(index.shape[0]>1):
                            second_pks += sout[i][pks[index[1:]]].sum()
                    else:
                        second_pks += sout[i][pks[index[0:]]].sum()
            amp_loss = -1*diag_out \
                + 0.5*second_pks
        if(use_mini_max_second_peaks):
            second_pks = 0
            # print(self.code_num)
            for i in range(self.code_num):
                pks, _ = find_peaks(sout[i].detach().numpy())
                values, index = sout[i][pks].abs().sort()
                if(index.shape[0]>=1):
                    if(pks[index[0]] == i):
                        if(index.shape[0]>1):
                            second_pks += sout[i][pks[index[1:]]].sum()
                    else:
                        second_pks += sout[i][pks[index[0:]]].sum()
            amp_loss = -1*diag_out \
                + 0.2*second_pks
        # print(self.need_var_gain)
        if(self.need_var_gain):
            out = amp_loss + var_loss
        else:
            out = amp_loss
        # out = amp_loss
        return out
    
    
class Model_W(Model_Basic):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, filename, device):
        super().__init__(filename, device)
        self.amp = nn.Parameter(torch.ones_like(self.w))
    def get_amp(self):
        amp = self.amp / torch.max(self.amp.abs())
        amp = amp.abs()
        amp = torch.clamp(amp, min=0.5)
        amp_half = torch.flipud(torch.fliplr(amp))
        return torch.vstack((amp, amp_half[1:]))
    
    def get_w(self):
        w_abs = self.w.abs()
        w_abs = torch.clamp(w_abs, min= 1)
        w = self.w / w_abs
        w_half = torch.flipud(torch.fliplr(w))
        return torch.vstack((w, w_half[1:])) * self.get_amp()
    
    def forward(self):
        w = self.get_w()
        # w = w * self.get_amp()
        theta = self.get_theta()
        lens_in = torch.matmul(w, self.G)
        lens_out = torch.matmul(lens_in, torch.diag(theta.squeeze()))
        sout = torch.matmul(lens_out, self.steerVec)
        
        sout = sout.abs() ** 2
        sout = sout / self.array_num / self.lens_num
        
        if(self.lens_type == "1D"):
            var_gain = 50
        elif(self.lens_type == "2D"):
            var_gain = 200
            
        diag_out = 0
        diag_sum = 0
        weights = self.get_weights().to(self.device)
        
        diag_out += torch.sum(torch.diag(sout, 0))
        
        var_loss = var_gain * torch.std(torch.diag(sout) / weights)
        
        use_sidelob_cancel = True
        
        if(use_sidelob_cancel):
            ignore_angle = 5
            for ti in range(-ignore_angle,ignore_angle+1):
                diag_sum += torch.sum(torch.diag(sout, ti))
                
            amp_loss = -diag_out + 0.05 * (torch.sum(sout) - diag_sum)
        


        out = amp_loss
        return out    




class Model_Side_lobe_gain(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, filename, device):
        super().__init__()
        # initialize weights with random numbers
        mat = scipy.io.loadmat(filename)
        # params = scipy.io.loadmat('optimal.mat')

        self.array_num = mat['G'].shape[0]
        self.code_num = mat['A'].shape[0]
        self.lens_num = mat['steerVec'].shape[0]
        self.device = device
        
        self.use_lagrange = False
        self.side_lobe_gain = 0

        
        
        # self.theta = torch.from_numpy(params['len_theta']).to(torch.cfloat).to(device)
        # self.theta = nn.Parameter(torch.from_numpy(params['len_theta']).to(torch.cfloat))
        
        # w_optimal = params['speaker_w']
        self.G = torch.from_numpy(mat['G']).to(torch.cfloat).to(device)
        self.steerVec = torch.from_numpy(mat['steerVec']).to(torch.cfloat).to(device)
        self.sweep_angle = torch.from_numpy(mat['sweep_angle']).to(device).view(-1)
        # self.add_weights = torch.from_numpy(mat['add_weights']).to(torch.cfloat).to(device)
        
        self.w = nn.Parameter(torch.rand(int((self.steerVec.shape[1]+1) / 2), self.array_num, dtype=torch.cfloat))
        
        self.lens_type = mat['lens_dimension']
        if(self.lens_type == "1D"):
            self.theta = nn.Parameter(torch.rand(int(self.lens_num / 2), 1, dtype=torch.cfloat))
            
            self.gain = 1 / self.lens_num
        elif(self.lens_type == "2D"):
            self.theta = nn.Parameter(torch.rand(int(self.lens_num / 4), 1, dtype=torch.cfloat))
            self.lens_width = int(np.sqrt(self.lens_num))
            self.gain = 1 / self.lens_num
        else:
            print("Wrong lens dimension.")
            return
            
        self.add_weights = torch.from_numpy(mat['add_weights']).to(torch.float).to(device)
        

    def get_w(self):
        # return self.w / torch.abs(self.w)
        # return 
        # w = self.w / torch.abs(self.w)
        # w = self.w / torch.max(self.w.abs(), dim=1, keepdim=True)[0]
        w_abs = self.w.abs()
        w_abs = torch.clamp(w_abs, min= 1)
        w = self.w / w_abs
        w_half = torch.flipud(torch.fliplr(w))
        return torch.vstack((w, w_half[1:]))

        # return self.w  / torch.sum(self.w.abs(), 1).unsqueeze(-1)
    def get_theta(self):
        if(self.lens_type == "1D"):
            theta = torch.vstack((self.theta, torch.flipud(self.theta)))
            with torch.no_grad():
                return theta / torch.abs(theta)
                # return theta
        elif(self.lens_type == "2D"):
            theta = torch.zeros(self.lens_width, self.lens_width, dtype=torch.cfloat).to(self.device)
            
            sub_mat_width = int(self.lens_width / 2)
            
            if(not self.use_lagrange):
                theta_squeeze = (self.theta / self.theta.abs()).view(sub_mat_width, -1)
            else:
                theta_squeeze = self.theta.view(sub_mat_width, -1)
            # test_index = torch.from_numpy(np.arange(64)).view(8,8)
            # target_index = torch.zeros(16, 16)
            
            for mi in range(sub_mat_width):
                theta[mi, 0:sub_mat_width] = theta_squeeze[mi]
                # target_index[mi, 0:8] = test_index[mi]
                theta[mi, sub_mat_width:self.lens_width] = torch.flipud(theta_squeeze[mi])
                # target_index[mi, 8:16] = torch.flipud(test_index[mi])
                theta[self.lens_width-1-mi] = theta[mi]
                # target_index[15-mi] = target_index[mi]
                
            return theta.view(-1)
    
    def get_weights(self):
        max_angle = 60
        atten = -0.2
        x = np.arange(-max_angle, max_angle+1)
        return atten / max_angle ** 2 * torch.from_numpy(x)**2 + 1
    
    def set_side_lobe_gain(self, side_lobe_gain):
        self.side_lobe_gain = side_lobe_gain
        

    def forward(self):
        w = self.get_w()
        theta = self.get_theta()
        # theta = self.get_theta()
        lens_in = torch.matmul(w, self.G)

        lens_out = torch.matmul(lens_in, torch.diag(theta.squeeze()))
        sout = torch.matmul(lens_out, self.steerVec)
        
        sout = sout.abs() ** 2
        sout = sout / self.array_num / self.lens_num
        
        if(self.lens_type == "1D"):
            # K = 0.01
            var_gain = 50
        elif(self.lens_type == "2D"):
            # K = 0.04
            var_gain = 200
        diag_out = 0
        diag_sum = 0
        weights = self.get_weights().to(self.device)
        
        diag_out += torch.sum(torch.diag(sout, 0))
        
        var_loss = var_gain * torch.std(torch.diag(sout) / weights)
        
        use_mini_max = False
        use_sidelob_cancel = False
        use_adaptive_minimax = False
        
        if(use_sidelob_cancel):
            ignore_angle = 5
            for ti in range(-ignore_angle,ignore_angle+1):
                diag_sum += torch.sum(torch.diag(sout, ti))
                
            # for ti in range(180 - 30, 180):
            #     # print(ti)
            #     diag_sum += torch.sum(torch.diag(sout, ti))
            
            # for ti in range(-180, -180+30):
            #     # print(ti)
            #     diag_sum += torch.sum(torch.diag(sout, ti))
                
            amp_loss = -diag_out \
            + self.side_lobe_gain*(torch.sum(sout) - diag_sum)
        
        if(use_mini_max):
            second_pks = 0
            # print(self.code_num)
            for i in range(self.code_num):
                pks, _ = find_peaks(sout[i].detach().numpy())
                values, index = sout[i][pks].abs().sort()
                if(index.shape[0]>=1):
                    if(pks[index[0]] == i):
                        if(index.shape[0]>1):
                            second_pks += sout[i][pks[index[1:]]].sum()
                    else:
                        second_pks += sout[i][pks[index[0:]]].sum()
            amp_loss = -1*diag_out \
                + 0.09*second_pks
            # print(second_pks.detach().numpy())
            # amp_loss += 2000*K*second_pks.sum()
            # print(20000000*second_pks)

        out = amp_loss + var_loss + 0*torch.mean(self.w.abs()**2)
        return out
    
class Model_Tune(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, filename, device):
        super().__init__()
        # initialize weights with random numbers
        mat = scipy.io.loadmat(filename)
        # params = scipy.io.loadmat('optimal.mat')

        self.array_num = mat['G'].shape[0]
        self.code_num = mat['A'].shape[0]
        self.lens_num = mat['steerVec'].shape[0]
        self.device = device
        
        self.use_lagrange = False
        self.G = torch.from_numpy(mat['G']).to(torch.cfloat).to(device)
        self.steerVec = torch.from_numpy(mat['steerVec']).to(torch.cfloat).to(device)
        self.sweep_angle = torch.from_numpy(mat['sweep_angle']).to(device).view(-1)
        
        self.w = nn.Parameter(torch.rand(int((self.steerVec.shape[1]+1)/2), self.array_num, dtype=torch.cfloat))
        
        self.lens_type = mat['lens_dimension']
        if(self.lens_type == "1D"):
            self.theta = nn.Parameter(torch.rand(int(self.lens_num / 2), 1, dtype=torch.cfloat))
            
            self.gain = 1 / self.lens_num
        elif(self.lens_type == "2D"):
            self.theta = nn.Parameter(torch.rand(int(self.lens_num / 4), 1, dtype=torch.cfloat))
            self.lens_width = int(np.sqrt(self.lens_num))
            self.gain = 1 / self.lens_num
        else:
            print("Wrong lens dimension.")
            return
        
        self.add_weights = torch.from_numpy(mat['add_weights']).to(torch.float).to(device)
        
        if(self.use_lagrange):
            self.lambda1 = nn.Parameter(torch.rand_like(self.theta)).to(device)
            self.lambda2 = nn.Parameter(torch.rand_like(self.w)).to(device)
            self.v1 = nn.Parameter(torch.rand_like(self.w)).to(device)

    def get_w(self):
        # return self.w / torch.abs(self.w)
        # return 
        # w = self.w / torch.max(self.w.abs(), dim=1, keepdim=True)[0]
        # with torch.no_grad():
        w_abs = self.w.abs()
        w_abs = torch.clamp(w_abs, min= 1)
        w = self.w / w_abs
        w_half = torch.flipud(torch.fliplr(w))
        return torch.vstack((w, w_half[1:]))
        # return self.w  / torch.sum(self.w.abs(), 1).unsqueeze(-1)
    def get_theta(self):
        if(self.lens_type == "1D"):
            theta = torch.vstack((self.theta, torch.flipud(self.theta)))
            if(not self.use_lagrange):
                with torch.no_grad():
                    return theta / torch.abs(theta)
            else:
                return theta
        elif(self.lens_type == "2D"):
            theta = torch.zeros(self.lens_width, self.lens_width, dtype=torch.cfloat).to(self.device)
            
            sub_mat_width = int(self.lens_width / 2)
            
            if(not self.use_lagrange):
                theta_squeeze = (self.theta / self.theta.abs()).view(sub_mat_width, -1)
            else:
                theta_squeeze = self.theta.view(sub_mat_width, -1)
            # test_index = torch.from_numpy(np.arange(64)).view(8,8)
            # target_index = torch.zeros(16, 16)
            
            for mi in range(sub_mat_width):
                theta[mi, 0:sub_mat_width] = theta_squeeze[mi]
                # target_index[mi, 0:8] = test_index[mi]
                theta[mi, sub_mat_width:self.lens_width] = torch.flipud(theta_squeeze[mi])
                # target_index[mi, 8:16] = torch.flipud(test_index[mi])
                theta[self.lens_width-1-mi] = theta[mi]
                # target_in dex[15-mi] = target_index[mi]
                
            return theta.view(-1)
    
    def get_weights(self):
        x = np.arange(-90, 91)
        return -0.6 / 90 ** 2 * torch.from_numpy(x)**2 + 1

    def forward(self):
        w = self.get_w()
        theta = self.get_theta()
           
        # theta = self.get_theta()
        lens_in = torch.matmul(w, self.G)

        lens_out = torch.matmul(lens_in, torch.diag(theta.squeeze()))
        sout = torch.matmul(lens_out, self.steerVec)
        
        sout = sout.abs() ** 2
        sout = sout / self.array_num / self.lens_num
        
        return -torch.sum(torch.diag(sout, 0))