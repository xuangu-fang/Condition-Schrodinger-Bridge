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
    prior = td.MultivariateNormal(torch.zeros(opt.data_dim), torch.eye(opt.data_dim[-1]))
    return PriorSampler(prior, batch_size, opt.device)

def build_data_sampler(opt, batch_size):
    return {
            'Scurve': Scurve,
            'Spiral': Spiral,
            'Moon': Moon,
            'Circle': Circle,
            'Pinwheel': Pinwheel,
        }.get(opt.problem_name)(batch_size)


def normalize(xs):
    return (xs - xs.mean()) / xs.std()

def normalize_both(xs):
    xs[:, 0] = normalize(xs[:, 0])
    xs[:, 1] = normalize(xs[:, 1])
    return xs 

""" motivated by GP-Sinkhorn's code """
class Scurve:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        return torch.Tensor(normalize(datasets.make_s_curve(n_samples=self.batch_size, noise=0.01)[0][:, [0, 2]]))
      

class Spiral:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        return torch.Tensor(normalize_both(datasets.make_swiss_roll(n_samples=self.batch_size, noise=0.01)[0][:, [0, 2]]))

class Moon:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        return torch.Tensor(normalize(datasets.make_moons(n_samples=self.batch_size, noise=noise)[0]))

class Circle:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        return torch.Tensor(normalize((datasets.make_circles(n_samples=self.batch_size, factor=0.4, noise=noise)[0])))

class Pinwheel:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
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
        samples = normalize(np.random.permutation(np.einsum("ti,tij->tj", features, rotations)))
        return torch.Tensor(samples)

class PriorSampler: # a dump prior sampler to align with DataSampler
    def __init__(self, prior, batch_size, device):
        self.prior = prior
        self.batch_size = batch_size
        self.device = device

    def log_prob(self, x):
        return self.prior.log_prob(x)

    def sample(self):
        return self.prior.sample([self.batch_size]).to(self.device)
