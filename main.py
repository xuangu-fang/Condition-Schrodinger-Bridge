from __future__ import absolute_import, division, print_function, unicode_literals
import os, time, gc

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage

import policy
import sde
import data
import util

from ipdb import set_trace as debug

import options
import colored_traceback.always

def build_optimizer_ema_sched(opt, policy):
    direction = policy.direction

    optim_name = {
        'Adam': Adam,
        'AdamW': AdamW,
        'Adagrad': Adagrad,
        'RMSprop': RMSprop,
        'SGD': SGD,
    }.get(opt.optimizer)

    optim_dict = {
            "lr": opt.lr_f if direction=='forward' else opt.lr_b,
            'weight_decay':opt.l2_norm,
    }
    if opt.optimizer == 'SGD':
        optim_dict['momentum'] = 0.9

    optimizer = optim_name(policy.parameters(), **optim_dict)
    ema = ExponentialMovingAverage(policy.parameters(), decay=0.99)
    if opt.lr_gamma < 1.0:
        sched = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
    else:
        sched = None

    return optimizer, ema, sched

def freeze_policy(policy):
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()
    return policy

def activate_policy(policy):
    for p in policy.parameters():
        p.requires_grad = True
    policy.train()
    return policy

""" functions from the original loss file """
def sample_gaussian_like(y):
    return torch.randn_like(y)

""" default Gaussian """
def sample_e(opt, x):
    return {
        'gaussian': sample_gaussian_like,
    }.get(opt.noise_type)(x)


def compute_div_gz(opt, dyn, ts, xs, policy, return_zs=False):

    """ Wei: feed the neural network with a long tensor (batch size x time steps) """
    zs = policy(xs, ts)
    """ Wei: dyn is dynamics from SDE (default ve_diffusions) """
    g_ts = dyn.g(ts)
    g_ts = g_ts[:, None]
    gzs = g_ts * zs
    e = sample_e(opt, xs)
    e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
    div_gz = e_dzdx * e

    return [div_gz, zs] if return_zs else div_gz

def compute_sb_nll_joint_train(opt, batch_x, dyn, ts, xs_f, zs_f, x_term_f, policy_b):
    """ Implementation of Eq (16) in our main paper.
    """
    assert opt.train_method == 'joint'
    assert policy_b.direction == 'backward'
    assert xs_f.requires_grad and zs_f.requires_grad and x_term_f.requires_grad

    div_gz_b, zs_b = compute_div_gz(opt, dyn, ts, xs_f, policy_b, return_zs=True)

    loss = 0.5*(zs_f + zs_b)**2 + div_gz_b
    """ loss so far is of shape [batch x time steps, 2]' """
    """ Wei: dyn.dt is a scalar of 0.01; batch_x is scaler of 512"""
    loss = torch.sum(loss*dyn.dt) / batch_x
    loss = loss - dyn.q.log_prob(x_term_f).mean()
    return loss

class Runner():
    def __init__(self, opt):
        super(Runner, self).__init__()

        self.start_time = time.time()

        self.ts = torch.linspace(opt.t0, opt.T, opt.interval)
        # build boundary distribution (p: target, q: prior)
        self.p, self.q = data.build_boundary_distribution(opt)
        # build dynamics, forward (z_f) and backward (z_b) policies
        self.dyn = sde.build(opt, self.p, self.q)
        self.z_f = policy.build(opt, self.dyn, 'forward')  # p -> q
        self.z_b = policy.build(opt, self.dyn, 'backward') # q -> p
        self.optimizer_f, self.ema_f, self.sched_f = build_optimizer_ema_sched(opt, self.z_f)
        self.optimizer_b, self.ema_b, self.sched_b = build_optimizer_ema_sched(opt, self.z_b)
        if opt.load:
            util.restore_checkpoint(opt, self, opt.load)
        if opt.log_tb: # tensorboard related things
            self.it_f = 0
            self.it_b = 0
            self.writer=SummaryWriter(
                log_dir=os.path.join('runs', opt.log_fn) if opt.log_fn is not None else None
            )


    def update_count(self, direction):
        if direction == 'forward':
            self.it_f += 1
            return self.it_f
        elif direction == 'backward':
            self.it_b += 1
            return self.it_b
        else:
            raise RuntimeError()

    def get_optimizer_ema_sched(self, z):
        if z == self.z_f:
            return self.optimizer_f, self.ema_f, self.sched_f
        elif z == self.z_b:
            return self.optimizer_b, self.ema_b, self.sched_b
        else:
            raise RuntimeError()

    def sb_joint_train(self, opt):
        policy_f, policy_b = self.z_f, self.z_b
        policy_f = activate_policy(policy_f)
        policy_b = activate_policy(policy_b)
        optimizer_f, _, sched_f = self.get_optimizer_ema_sched(policy_f)
        optimizer_b, _, sched_b = self.get_optimizer_ema_sched(policy_b)

        ts      = self.ts
        batch_x = opt.samp_bs

        for it in range(opt.num_itr):

            optimizer_f.zero_grad()
            optimizer_b.zero_grad()
            xs_f, zs_f, x_term_f = self.dyn.sample_traj(ts, policy_f, save_traj=True)
            xs_f = util.flatten_dim01(xs_f)
            zs_f = util.flatten_dim01(zs_f)
            _ts = ts.repeat(batch_x)
            loss = compute_sb_nll_joint_train(
                opt, batch_x, self.dyn, _ts, xs_f, zs_f, x_term_f, policy_b
            )
            loss.backward()
            optimizer_f.step()
            optimizer_b.step()

            if sched_f is not None: sched_f.step()
            if sched_b is not None: sched_b.step()
            self.log_sb_joint_train(opt, it, loss, optimizer_f, opt.num_itr)
            # evaluate
            if (it + 1) % opt.eval_itr == 0:
                with torch.no_grad():
                    xs_b, _, _ = self.dyn.sample_traj(ts, policy_b, save_traj=True)
                util.save_toy_npy_traj(opt, 'train_{}_lr_{}_sigma_{}_node_{}_block_{}_it{}_seed_{}'.format(opt.problem_name, opt.lr, opt.sigma_max, opt.hidden_nodes, opt.blocks, it+1, opt.seed), xs_b.detach().cpu().numpy())

    def _print_train_itr(self, it, loss, optimizer, num_itr, name):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        print("[{0}] train_it {1}/{2} | lr:{3} | loss:{4} | time:{5}"
            .format(
                util.magenta(name),
                util.cyan("{}".format(1+it)),
                num_itr,
                util.yellow("{:.2e}".format(lr)),
                util.red("{:.4f}".format(loss.item())),
                util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))
    def log_sb_joint_train(self, opt, it, loss, optimizer, num_itr):
        self._print_train_itr(it, loss, optimizer, num_itr, name='SB joint')


print(util.yellow("======================================================="))
print(util.yellow("     Likelihood-Training of Schrodinger Bridge"))
print(util.yellow("======================================================="))
print(util.magenta("setting configurations..."))
opt = options.set()

def main(opt):
    run = Runner(opt)

    # ====== Only joint training functions ======
    run.sb_joint_train(opt)


if __name__ == "__main__":
    if not opt.cpu:
        with torch.cuda.device(opt.gpu):
            main(opt)
    else: 
        main(opt)
