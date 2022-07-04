import os,sys,re

import numpy as np
import shutil
import termcolor
import pathlib
from scipy import linalg
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.utils as tu
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from ipdb import set_trace as debug


# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_time(sec):
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = sec%60
    return h,m,s

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def flatten_dim01(x):
    # (dim0, dim1, *dim2) --> (dim0x1, *dim2)
    return x.reshape(-1, *x.shape[2:])

def unflatten_dim01(x, dim01):
    # (dim0x1, *dim2) --> (dim0, dim1, *dim2)
    return x.reshape(*dim01, *x.shape[1:])

def compute_z_norm(zs, dt):
    # Given zs.shape = [batch, timesteps, *z_dim], return E[\int 0.5*norm(z)*dt],
    # where the norm is taken over z_dim, the integral is taken over timesteps,
    # and the expectation is taken over batch.
    zs = zs.reshape(*zs.shape[:2],-1)
    return 0.5 * zs.norm(dim=2).sum(dim=1).mean(dim=0) * dt

def save_toy_npy_traj(opt, fn, traj, n_snapshot=None, direction=None):
    #form of traj: [bs, interval, x_dim=2]
    fn_npy = os.path.join('results', opt.dir, fn+'.npy')
    fn_pdf = os.path.join('results', opt.dir, fn+'.pdf')

    lims = [-2.6, 2.6]

    if n_snapshot is None: # only store t=0
        plt.scatter(traj[:,0,0],traj[:,0,1], s=5)
        plt.xlim(*lims)
        plt.ylim(*lims)
    else:
        total_steps = traj.shape[1]
        sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)
        fig, axs = plt.subplots(1, n_snapshot)
        fig.set_size_inches(n_snapshot*6, 6)
        color = 'salmon' if direction=='forward' else 'royalblue'
        for ax, step in zip(axs, sample_steps):
            ax.scatter(traj[:,step,0],traj[:,step,1], s=5, color=color)
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)
            ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
        fig.tight_layout()

    plt.savefig(fn_pdf)
    np.save(fn_npy, traj)
    plt.clf()
