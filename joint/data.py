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

    opt.data_dim = [2]
    prior = build_prior_sampler(opt, opt.samp_bs)
    pdata = build_data_sampler(opt, opt.samp_bs)

    return pdata, prior

def build_prior_sampler(opt, batch_size):
    prior = td.MultivariateNormal(torch.zeros(opt.data_dim).to(opt.device), torch.eye(opt.data_dim[-1]).to(opt.device))
    return PriorSampler(prior, batch_size, opt.device)

def build_data_sampler(opt, batch_size):
    return {
            'Scurve': Scurve,
            'Spiral': Spiral,
            'Moon': Moon,
            'Circle': Circle,
            'Pinwheel': Pinwheel,
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

""" Scurve_condi"""
class Scurve_condi:
    def __init__(self, batch_size, device,mask_ratio=0.1):
        self.batch_size = batch_size
        self.device = device
        # self.samples = normalize(datasets.make_s_curve(n_samples=self.batch_size, noise=0.01)[0][:, [0, 2]])

        # right now, just assume we observe all samples of gruonded truth
        x_obs = normalize(datasets.make_s_curve(n_samples=self.batch_size, noise=0.01)[0][:, [0, 2]])

        # sort it by y-value, convinient for visulization 
        sort_arg = np.argsort(x_obs[:,1])
        new_x = x_obs[sort_arg,:]

        self.x_obs = new_x
        self.samples = self.x_obs

        self.mask_ratio = mask_ratio
        self.num_x_condi= round(self.mask_ratio * self.batch_size)

        # update when do sampling
        self.x_condi = None
        self.mask_target = None

        # consistent with CSDI data class, to be further set
        self.mask_gt = None
        self.mask_obs = None
        self.mask_valid = None # for validation and visulization 

        self.update_mask()

    def update_mask(self):
        shuffel_ind = np.random.permutation(self.batch_size)
        idx_condi = shuffel_ind[:self.num_x_condi]
        idx_target = shuffel_ind[self.num_x_condi:]

        mask_condi = np.zeros_like(self.x_obs)
        mask_condi[idx_condi,:]=1

        x_target = np.multiply(self.x_obs,1-mask_condi)
        x_condi = np.multiply(self.x_obs,mask_condi)

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
      

class Spiral:
    def __init__(self, batch_size, device):
        self.device = device
        self.batch_size = batch_size
        self.samples = normalize_both(datasets.make_swiss_roll(n_samples=self.batch_size, noise=0.01)[0][:, [0, 2]])

    def sample(self):
        return torch.Tensor(self.samples).to(self.device)

class Moon:
    def __init__(self, batch_size, device):
        self.device = device
        self.batch_size = batch_size
        self.samples = normalize(datasets.make_moons(n_samples=self.batch_size, noise=0)[0])

    def sample(self):
        return torch.Tensor(self.samples).to(self.device)

class Circle:
    def __init__(self, batch_size, device):
        self.device = device
        self.batch_size = batch_size
        self.samples = normalize((datasets.make_circles(n_samples=self.batch_size, factor=0.4, noise=0)[0]))

    def sample(self):
        return torch.Tensor(self.samples).to(self.device)

class Pinwheel:
    def __init__(self, batch_size, device):
        self.device = device
        self.batch_size = batch_size
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = self.batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = (np.random.randn(num_classes * num_per_class, 2) *
                    np.array([radial_std, tangential_std]))
        features[:, 0] += 1
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        self.samples = normalize(np.random.permutation(np.einsum("ti,tij->tj", features, rotations)))
               
    def sample(self):
        return torch.Tensor(self.samples).to(self.device)

class PriorSampler: # a dump prior sampler to align with DataSampler
    def __init__(self, prior, batch_size, device):
        self.device = device
        self.prior = prior
        self.batch_size = batch_size
        

    def log_prob(self, x):
        return self.prior.log_prob(x)

    def sample(self):
        return self.prior.sample([self.batch_size]).to(self.device)
