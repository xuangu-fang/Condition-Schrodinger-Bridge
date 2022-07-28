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
    pdata = build_data_sampler(opt, opt.samp_bs)

    return pdata, prior

def build_prior_sampler(opt, batch_size):
    prior = td.MultivariateNormal(torch.zeros(opt.data_dim).to(opt.device), torch.eye(opt.data_dim[-1]).to(opt.device))
    return PriorSampler(prior, batch_size, opt.device)

def build_data_sampler(opt, batch_size):
    return {
            'Scurve': Scurve,
            'Sin':Sin,
            'Scurve_condi':Scurve_condi
        }.get(opt.problem_name)(batch_size,opt.device)


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

def _make_sin(batch,L = 50):
    x = np.linspace(0,1,L)
    y = np.sin(x*2*np.pi)
    y = y.reshape(1,-1).repeat(batch,0) 
    y = y + np.random.randn(*y.shape) * 0.1

    return x,y


class Sin:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        # self.samples = normalize(datasets.make_s_curve(n_samples=self.batch_size, noise=0.01)[0][:, [0, 2]])
        _, self.samples = _make_sin(batch_size)
        self.device = device
    def sample(self):
        return torch.Tensor(self.samples).to(self.device)

""" Scurve_condi"""
class Scurve_condi:
    def __init__(self, batch_size, device,mask_ratio=0.1):
        self.batch_size = batch_size
        self.device = device
        # self.samples = normalize(datasets.make_s_curve(n_samples=self.batch_size, noise=0.01)[0][:, [0, 2]])

        # right now, just assume we observe all samples of gruonded truth

        # x_obs = normalize(datasets.make_s_curve(n_samples=self.batch_size, noise=0.01)[0][:, [0, 2]])
        x_obs = normalize(datasets.make_s_curve(n_samples=1000, noise=0.01)[0][:, [0, 2]])


        # sort it by y-value, convinient for visulization 
        sort_arg = np.argsort(x_obs[:,1])
        new_x = x_obs[sort_arg,:]


        self.x_obs = new_x[:self.batch_size,:]
        self.samples = self.x_obs

        self.mask_ratio = mask_ratio
        self.num_x_condi= round(self.mask_ratio * self.batch_size)

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
        idx_test =  np.concatenate([np.arange(0,250),np.arange(400,600)])
        mask_test = np.zeros_like(self.x_obs)
        mask_test[idx_test]=1

        self.mask_test = mask_test


    def update_mask(self):
        ''' for fast test start'''
        # idx_condi = np.concatenate([np.arange(0,100),np.arange(450,550)])
        # idx_target =  np.concatenate([np.arange(100,450),np.arange(550,self.batch_size)])
        # mask_condi = np.zeros_like(self.x_obs)
        # mask_condi[idx_condi,:]=1
        # mask_target = 1-mask_condi

        # x_target = np.multiply(self.x_obs,1-mask_condi)
        # x_condi = np.multiply(self.x_obs,mask_condi)
        ''' for fast test end'''
        shuffel_ind = np.random.permutation(self.batch_size)
        idx_condi = shuffel_ind[:self.num_x_condi]
        idx_target = shuffel_ind[self.num_x_condi:]

        mask_condi = np.zeros_like(self.x_obs)
        mask_condi[idx_condi,:]=1

        x_target = np.multiply(self.x_obs,1-mask_condi)
        x_condi = np.multiply(self.x_obs,mask_condi)

        x_target = np.multiply(x_target,1-self.mask_test)
        # x_condi = np.multiply(x_condi,1-self.mask_test)

        self.x_condi = torch.Tensor(x_condi).to(self.device)
        self.x_target = torch.Tensor(x_target).to(self.device)

        self.mask_target = torch.Tensor(1-mask_condi).to(self.device)
        self.idx_target = idx_target
        self.idx_condi = idx_condi




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
    def __init__(self, prior, batch_size, device):
        self.device = device
        self.prior = prior
        self.batch_size = batch_size
        

    def log_prob(self, x):
        return self.prior.log_prob(x)

    def sample(self):
        return self.prior.sample([self.batch_size]).to(self.device)
