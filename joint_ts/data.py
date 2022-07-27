import numpy as np

import torch
import torch.distributions as td
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from prefetch_generator import BackgroundGenerator

import util
from ipdb import set_trace as debug

from sklearn import datasets


def build_boundary_distribution(opt):
    print(util.magenta("build boundary distribution..."))

    # opt.data_dim = [2]
    # pdata = build_data_sampler(opt, opt.samp_bs)

    pdata = TS_condi_simu(batch_size = opt.samp_bs,device = opt.device, mask_ratio=0.2)

    prior = build_prior_sampler(opt, opt.samp_bs,pdata.K,pdata.L)
    
    return pdata, prior

def build_prior_sampler(opt, batch_size, K,L):
    # total_size = K*L
    cov_coef = opt.sigma_max**2
    prior = td.MultivariateNormal(torch.zeros(opt.data_dim).to(opt.device), cov_coef* torch.eye(opt.data_dim[-1]).to(opt.device))
    return PriorSampler(prior, batch_size, opt.device, K,L)


def normalize(xs):
    return (xs - xs.mean()) / xs.std()

def normalize_both(xs):
    xs[:, 0] = normalize(xs[:, 0])
    xs[:, 1] = normalize(xs[:, 1])
    return xs 

""" motivated by GP-Sinkhorn's code """
class Scurve:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.samples = normalize(datasets.make_s_curve(n_samples=self.batch_size, noise=0.01)[0][:, [0, 2]])
        self.device = device
    def sample(self):
        return torch.Tensor(self.samples).to(self.device)


""" class for multi-var time-series imputation on simulation data"""
class TS_condi_simu:
    def __init__(self, batch_size, device,mask_ratio=0.5):
        self.batch_size = batch_size
        self.device = device

        data_dict = np.load('./data/simu_data_noise.npy',allow_pickle=True)

        self.time_step = data_dict.item().get('time_step') # the real time points of time series 
        self.data = data_dict.item().get('data_all') # 8*100

        self.data = self.data[0:2,:]# 2*100

        # self.mask_train = data_dict.item().get('mask_train')

        self.K,self.L = self.data.shape

        # self.x_obs = self
        # self.samples = self.x_obs

        self.mask_ratio = mask_ratio
        self.num_x_condi= round(self.mask_ratio * self.L * self.K)

        # update when do sampling
        self.x_condi = None
        self.mask_target = None

        # consistent with CSDI data class, to be further set
        self.mask_test = None
        self.mask_obs = None
        self.mask_valid = None # for validation and visulization 

        self.generate_mask_test()
        self.update_mask()


    def generate_mask_test(self):
        # the training of value at mask_test is always non-acessable for training
        idx_test =  np.concatenate([np.arange(10,30),np.arange(70,80)])
        mask_test = np.zeros_like(self.data)
        mask_test[:,idx_test]=1

        self.mask_test = mask_test
        self.mask_obs = 1-self.mask_test


    def update_mask(self):
        '''update the mask_condi and mask_target on the obseved(training) data during each sampling'''

        x_condi_list = []
        x_target_list = []
        
        mask_condi_list = []
        mask_target_list = []

        obseved_tp_list = []

        for i in range(self.batch_size):

            shuffel_ind = np.random.permutation(self.L*self.K)
            idx_condi = shuffel_ind[:self.num_x_condi]
            idx_target = shuffel_ind[self.num_x_condi:]

            mask_condi = np.zeros(self.L*self.K)
            mask_condi[idx_condi]=1

            mask_condi = mask_condi.reshape(self.K,self.L)
            mask_target = 1-mask_condi

            # fang: for real sampling, here should be self.data + noise or from real dataset 

            x_target = np.multiply(self.data,mask_target)
            x_condi = np.multiply(self.data,mask_condi)

            x_condi_list.append(x_condi)
            x_target_list.append(x_target)
            mask_condi_list.append(mask_condi)
            mask_target_list.append(mask_target)

            # fang: for real dataset, here should dependents of sample/sub-sequence
            obseved_tp_list.append(self.time_step)



        x_condi = np.stack(x_condi_list,0) # (B,K,L)
        x_target = np.stack(x_target_list,0) # (B,K,L)

        mask_condi = np.stack(mask_condi_list,0) # (B,K,L)
        mask_target = np.stack(mask_target_list,0) # (B,K,L)

        obseved_tp = np.stack(obseved_tp_list,0) # (B,L)


        self.x_condi = torch.Tensor(x_condi).to(self.device)
        self.x_target = torch.Tensor(x_target).to(self.device)

        self.mask_condi = torch.Tensor(mask_condi).to(self.device)
        self.mask_target = torch.Tensor(mask_target).to(self.device)
        
        self.observed_tp = torch.Tensor(obseved_tp).to(self.device)

        # self.idx_target = idx_target
        # self.idx_condi = idx_condi

    def sample(self,update=False):
        # when do sampling, we actually sample from the observations, 
        # and randomly split it as x_target, and x_conditional  

        if update:    
            self.update_mask()

        else:
            pass

        return self.x_target
        # return torch.Tensor(x_target).to(self.device)
      


class PriorSampler: # a dump prior sampler to align with DataSampler
    def __init__(self, prior, batch_size, device,K,L):
        self.device = device
        self.prior = prior
        self.batch_size = batch_size
        self.K = K
        self.L = L

    def log_prob(self, x):
        return self.prior.log_prob(x).mean(-1) # size: [B]

    def sample(self):
        # return self.prior.sample([self.batch_size]).reshape(self.batch_size,self.K,self.L).to(self.device)# (B,K,L)
        return self.prior.sample([self.batch_size]).to(self.device)# (B,K,L)

