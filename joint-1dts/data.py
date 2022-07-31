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
    prior = build_prior_sampler(opt, opt.samp_bs)
    pdata = build_data_sampler(opt)

    return pdata, prior

def build_prior_sampler(opt, batch_size):
    cov_coef = 1
    prior = td.MultivariateNormal(torch.zeros(opt.data_dim).to(opt.device), cov_coef*torch.eye(opt.data_dim[-1]).to(opt.device))
    return PriorSampler(prior, batch_size, opt.device)

def build_data_sampler(opt):
    return {
            'Scurve': Scurve,
            'Sin':Sin,
            'Sin-Condi':Sin_condi
        }.get(opt.problem_name)(opt)


def normalize(xs):
    return (xs - xs.mean()) / xs.std()

def normalize_both(xs):
    xs[:, 0] = normalize(xs[:, 0])
    xs[:, 1] = normalize(xs[:, 1])
    return xs 

""" motivated by GP-Sinkhorn's code """
class Scurve:
    def __init__(self, opt):
        self.batch_size = opt.samp_bs
        self.samples = normalize(datasets.make_s_curve(n_samples=self.batch_size, noise=0.01)[0][:, [0, 2]])
        self.device = opt.device
    def sample(self):
        return torch.Tensor(self.samples).to(self.device)

def _make_sin(batch,L = 50):
    x = np.linspace(0,1,L)
    y = np.sin(x*2*np.pi)
    y = y.reshape(1,-1).repeat(batch,0) 
    y = y + np.random.randn(*y.shape) * 0.001

    return x,y


class Sin:
    def __init__(self, opt):
        self.batch_size = opt.samp_bs
        # self.samples = normalize(datasets.make_s_curve(n_samples=self.batch_size, noise=0.01)[0][:, [0, 2]])
        _, self.samples = _make_sin(self.batch_size,opt.data_dim[0])
        self.device = opt.device
    def sample(self):
        return torch.Tensor(self.samples).to(self.device)

class Sin_condi:
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = opt.samp_bs
        _, self.data = _make_sin(1,opt.data_dim[0])
        self.data = self.data.squeeze() # (L)
        self.device = opt.device

        self.mask_ratio = opt.mask_ratio
        self.data_size = self.opt.data_dim[0]
        self.num_x_condi= round(self.mask_ratio * self.data_size)

        # update when do sampling
        self.x_condi = None
        self.x_target = None
        self.mask_target = None

        # consistent with CSDI data class, to be further set
        self.mask_test = None
        self.mask_obs = None
        self.mask_valid = None # for validation and visulization 

        self.generate_mask_test()
        self.update_mask()

    def generate_mask_test(self):
        # the training of value at mask_test is always non-acessable for training
        # idx_test =  np.concatenate([np.arange(10,13),np.arange(30,33),np.arange(20,23)])
        idx_test =  np.concatenate([np.arange(20,30)])

        mask_test = np.zeros_like(self.data)
        mask_test[idx_test]=1

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

            shuffel_ind = np.random.permutation(self.data_size)
            idx_condi = shuffel_ind[:self.num_x_condi]
            idx_target = shuffel_ind[self.num_x_condi:]

            mask_condi = np.zeros(self.data_size)
            mask_condi[idx_condi]=1

            # mask_condi = mask_condi.reshape(self.K,self.L)
            mask_target = 1-mask_condi

            x_target = np.multiply(self.data,mask_target)
            x_condi = np.multiply(self.data,mask_condi)

            x_condi_list.append(x_condi)
            x_target_list.append(x_target)
            mask_condi_list.append(mask_condi)
            mask_target_list.append(mask_target)

            # fang: for real dataset, here should dependents of sample/sub-sequence
            # obseved_tp_list.append(self.time_step)



        x_condi = np.stack(x_condi_list,0) # (B,*)
        x_target = np.stack(x_target_list,0) # (B,*)

        mask_condi = np.stack(mask_condi_list,0) # (B,*)
        mask_target = np.stack(mask_target_list,0) # (B,*)

        # obseved_tp = np.stack(obseved_tp_list,0) # (B,L)


        self.x_condi = torch.Tensor(x_condi).to(self.device)
        self.x_target = torch.Tensor(x_target).to(self.device)

        self.mask_condi = torch.Tensor(mask_condi).to(self.device)
        self.mask_target = torch.Tensor(mask_target).to(self.device)
        
        # self.observed_tp = torch.Tensor(obseved_tp).to(self.device)

    def sample(self,update=False):
        # when do sampling, we actually sample from the observations, 
        # and randomly split it as x_target, and x_conditional  

        if update:    
            self.update_mask()

        else:
            pass

        # _, self.samples = _make_sin(self.batch_size,self.opt.data_dim[0])

        return self.x_target
        # return torch.Tensor(self.samples).to(self.device)

      

class PriorSampler: # a dump prior sampler to align with DataSampler
    def __init__(self, prior, batch_size, device):
        self.device = device
        self.prior = prior
        self.batch_size = batch_size
        

    def log_prob(self, x):
        return self.prior.log_prob(x)

    def sample(self):
        return self.prior.sample([self.batch_size]).to(self.device)
