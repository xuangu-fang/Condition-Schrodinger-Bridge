import numpy as np
import abc
from tqdm import tqdm
from functools import partial
import torch

import util
from ipdb import set_trace as debug

def _assert_increasing(name, ts):
    assert (ts[1:] > ts[:-1]).all(), '{} must be strictly increasing'.format(name)

def build(opt, p, q):
    print(util.magenta("build base sde..."))

    return {
        've': VESDE,
    }.get(opt.sde_type)(opt, p, q)


class BaseSDE(metaclass=abc.ABCMeta):
    def __init__(self, opt, p, q):
        self.opt = opt
        self.dt=opt.T / opt.interval
        self.p = p # data distribution
        self.q = q # prior distribution

        # set/update only when sample from p (forward) 
        self.x_condi = None 
        self.mask_target = None 

        # fang: these two vars will be updated after sampling,
        #  and be used to build side_info for both forward and backward
        self.mask_condi = None 
        self.observed_tp = None

        # self.idx_target = None
        # self.idx_condi = None 
        # self.side_info = None # follow CSDI, side_info = location embed + mask_condi



    @abc.abstractmethod
    def _f(self, x, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _g(self, x, t):
        raise NotImplementedError

    def f(self, x, t, direction):
        sign = 1. if direction=='forward' else -1.
        return sign * self._f(x,t)

    def g(self, t):
        return self._g(t)

    def dw(self, x, dt=None):
        dt = self.dt if dt is None else dt
        return torch.randn_like(x)*np.sqrt(dt)

    def propagate(self, t, x, z, direction, f=None, dw=None, dt=None):
        g = self.g(  t)
        f = self.f(x,t,direction) if f is None else f
        dt = self.dt if dt is None else dt
        dw = self.dw(x,dt) if dw is None else dw

        return x + (f + g*z)*dt + g*dw

    def sample_traj(self, ts, policy, corrector=None, apply_trick=True, save_traj=True, update_mask = False):

        # first we need to know whether we're doing forward or backward sampling
        opt = self.opt
        direction = policy.direction
        assert direction in ['forward','backward']

        # set up ts and init_distribution
        _assert_increasing('ts', ts)
        # init_dist = self.p if direction=='forward' else self.q
        # ts = ts if direction=='forward' else torch.flip(ts, dims=[0])
        # x= init_dist.sample() # [bs, x_dim]

        if direction=='forward':
            init_dist = self.p
            ts = ts

            # if it's a condi-dist, 
            # we also update its x_condi and mask_target/cond during the sampling

            x = init_dist.sample(update_mask) 

            if hasattr(init_dist, 'x_condi') and hasattr(init_dist, 'mask_target'):
                self.x_condi = init_dist.x_condi
                self.mask_target = init_dist.mask_target

                self.mask_condi = init_dist.mask_condi
                self.observed_tp = init_dist.observed_tp

                self.mask_train = 1-torch.Tensor(init_dist.mask_test).to(opt.device)

                # self.side_info = init_dist.side_info

        else:
            init_dist = self.q
            ts = torch.flip(ts, dims=[0])
            x = init_dist.sample()

        policy.net.set_x_condi(self.x_condi)
        policy.net.set_side_info(self.observed_tp, self.mask_condi)


        xs = torch.empty((x.shape[0], len(ts), *x.shape[1:])) if save_traj else None
        zs = torch.empty_like(xs) if save_traj else None
        # don't use tqdm for fbsde since it'll resample every itr
        _ts = ts if opt.train_method=='joint' else tqdm(ts,desc=util.yellow("Propagating Dynamics..."))
        for idx, t in enumerate(_ts):
            _t=t if idx==ts.shape[0]-1 else ts[idx+1]

            f = self.f(x,t,direction)
            z = policy(x,t)
            dw = self.dw(x)

            t_idx = idx if direction=='forward' else len(ts)-idx-1
            if save_traj:
                xs[:,t_idx,...]=x
                zs[:,t_idx,...]=z

            x = self.propagate(t, x, z, direction, f=f, dw=dw)

        x_term = x

        res = [xs, zs, x_term]
        return res
    
""" default setups """
class VESDE(BaseSDE):
    def __init__(self, opt, p, q):
        super(VESDE,self).__init__(opt, p, q)
        self.s_min=opt.sigma_min
        self.s_max=opt.sigma_max

    def _f(self, x, t):
        return torch.zeros_like(x)

    def _g(self, t):
        return compute_ve_diffusion(t, self.s_min, self.s_max)

####################################################
##  Implementation of SDE analytic kernel         ##
##  Ref: https://arxiv.org/pdf/2011.13456v2.pdf,  ##
##       page 15-16, Eq (30,32,33)                ##
####################################################

def compute_sigmas(t, s_min, s_max):
    return s_min * (s_max/s_min)**t

def compute_ve_g_scale(s_min, s_max):
    return np.sqrt(2*np.log(s_max/s_min))

def compute_ve_diffusion(t, s_min, s_max):
    return compute_sigmas(t, s_min, s_max) * compute_ve_g_scale(s_min, s_max)
