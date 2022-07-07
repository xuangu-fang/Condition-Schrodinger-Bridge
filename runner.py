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

''' ==============================================================================
    ===============   Starting module of alternative trainig loss ================
    ============================================================================== '''

def compute_sb_nll_alternate_train(opt, dyn, ts, xs, zs_impt, policy_opt, return_z=False):
    """ Implementation of Eq (18, 19) in our main paper.
    """
    assert opt.train_method == 'alternate'
    assert xs.requires_grad
    assert not zs_impt.requires_grad

    batch_x = opt.train_bs_x
    batch_t = opt.train_bs_t

    with torch.enable_grad():
        div_gz, zs = compute_div_gz(opt, dyn, ts, xs, policy_opt, return_zs=True)
        loss = zs*(0.5*zs + zs_impt) + div_gz
        loss = torch.sum(loss * dyn.dt) / batch_x / batch_t  # sum over x_dim and T, mean over batch
    return loss, zs if return_z else loss

''' ==============================================================================
    ================   Ending module of alternative trainig loss =================
    ============================================================================== '''


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
    loss = torch.sum(loss * dyn.dt) / batch_x
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

    ''' ==============================================================================
        ====================   Starting module of alternative trainig ================
        ============================================================================== '''

    def sample_train_data(self, opt, policy_opt, policy_impt, reused_sampler):
        train_ts = self.ts

        # reuse or sample training xs and zs
        try:
            reused_traj = next(reused_sampler)
            train_xs, train_zs = reused_traj[:,0,...], reused_traj[:,1,...]
            print('generate train data from [{}]!'.format(util.green('reused samper')))
        except:
            _, ema, _ = self.get_optimizer_ema_sched(policy_opt)
            _, ema_impt, _ = self.get_optimizer_ema_sched(policy_impt)
            with ema.average_parameters(), ema_impt.average_parameters():
                policy_impt = freeze_policy(policy_impt)
                policy_opt  = freeze_policy(policy_opt)
                """ =====================================================================================
                    =============== Caution !!!!! I ignored Langevin corrector at this moment ===========
                    ===================================================================================== """
                corrector = None # (lambda x,t: policy_impt(x,t) + policy_opt(x,t)) if opt.use_corrector else None
                xs, zs, _ = self.dyn.sample_traj(train_ts, policy_impt, corrector=corrector)
                train_xs = xs.detach().cpu(); del xs
                train_zs = zs.detach().cpu(); del zs
            print('generate train data from [{}]!'.format(util.red('sampling')))

        assert train_xs.shape[0] == opt.samp_bs
        assert train_xs.shape[1] == len(train_ts)
        assert train_xs.shape == train_zs.shape
        """ ===========================================================================
            ================   wei's comment: need to figure it out  ==================
            =========================================================================== """
        gc.collect()

        return train_xs, train_zs, train_ts

    def sb_alternate_train_stage(self, opt, stage, epoch, direction, reused_sampler=None):
        policy_opt, policy_impt = {
            'forward':  [self.z_f, self.z_b], # train forwad,   sample from backward
            'backward': [self.z_b, self.z_f], # train backward, sample from forward
        }.get(direction)

        for ep in range(epoch):
            # prepare training data
            train_xs, train_zs, train_ts = self.sample_train_data(
                opt, policy_opt, policy_impt, reused_sampler
            )

            # train one epoch
            policy_impt = freeze_policy(policy_impt)
            policy_opt = activate_policy(policy_opt)
            self.sb_alternate_train_ep(
                opt, ep, stage, direction, train_xs, train_zs, train_ts, policy_opt, epoch
            )

    def sb_alternate_train_ep(
        self, opt, ep, stage, direction, train_xs, train_zs, train_ts, policy, num_epoch
    ):
        assert train_xs.shape[0] == opt.samp_bs
        assert train_zs.shape[0] == opt.samp_bs
        assert train_ts.shape[0] == opt.interval
        assert direction == policy.direction

        optimizer, ema, sched = self.get_optimizer_ema_sched(policy)

        for it in range(opt.num_itr):
            # -------- sample x_idx and t_idx \in [0, interval] --------
            samp_x_idx = torch.randint(opt.samp_bs,  (opt.train_bs_x,))
            samp_t_idx = torch.randint(opt.interval, (opt.train_bs_t,))
            if opt.use_arange_t: samp_t_idx = torch.arange(opt.interval)

            # -------- build sample --------
            ts = train_ts[samp_t_idx].detach()
            xs = train_xs[samp_x_idx][:, samp_t_idx, ...].to(opt.device)
            zs_impt = train_zs[samp_x_idx][:, samp_t_idx, ...].to(opt.device)

            optimizer.zero_grad()
            xs.requires_grad_(True) # we need this due to the later autograd.grad

            # -------- handle for batch_x and batch_t ---------
            # (batch, T, xdim) --> (batch*T, xdim)
            xs      = util.flatten_dim01(xs)
            zs_impt = util.flatten_dim01(zs_impt)
            ts = ts.repeat(opt.train_bs_x)
            assert xs.shape[0] == ts.shape[0]
            assert zs_impt.shape[0] == ts.shape[0]

            # -------- compute loss and backprop --------
            loss, zs = compute_sb_nll_alternate_train(
                opt, self.dyn, ts, xs, zs_impt, policy, return_z=True
            )
            assert not torch.isnan(loss)

            loss.backward()

            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm(policy.parameters(), opt.grad_clip)
            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            
            # -------- logging --------
            zs = util.unflatten_dim01(zs, [len(samp_x_idx), len(samp_t_idx)])
            zs_impt = zs_impt.reshape(zs.shape)
            self.log_sb_alternate_train(
                opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch
            )

    def sb_alternate_train(self, opt):
        for stage in range(opt.num_stage):
            forward_ep = backward_ep = opt.num_epoch

            # train backward policy;
            # skip the trainining at first stage if checkpoint is loaded
            train_backward = not (stage == 0 and opt.load is not None)
            if train_backward:
                self.sb_alternate_train_stage(opt, stage, backward_ep, 'backward')

            # evaluate backward policy;
            # reuse evaluated trajectories for training forward policy
            n_reused_trajs = 0 # forward_ep * opt.samp_bs if opt.reuse_traj else 0 # 
            reused_sampler = self.evaluate(opt, stage+1, n_reused_trajs=n_reused_trajs)

            # train forward policy
            self.sb_alternate_train_stage(
                opt, stage, forward_ep, 'forward', reused_sampler=reused_sampler
            )


    @torch.no_grad()
    def evaluate(self, opt, stage, n_reused_trajs=0, metrics=None):
        #if util.is_toy_dataset(opt): # yes you are toy
        SYNTAX = f'{opt.problem_name}_lr_{opt.lr:.0e}_{opt.lr_gamma}_L_{opt.blocks}_B_{opt.samp_bs}_e_{opt.num_itr}'
        if 1:
            _, snapshot, ckpt = util.evaluate_stage(opt, stage, metrics=None)
            if snapshot:
                for z in [self.z_f, self.z_b]:
                    z = freeze_policy(z)
                    xs, _, _ = self.dyn.sample_traj(self.ts, z, save_traj=True)

                    fn = f"{SYNTAX}_s{stage}-{z.direction[0]}"
                    util.save_toy_npy_traj(
                        opt, fn, xs.detach().cpu().numpy(), n_snapshot=15, direction=z.direction
                    )
                    #util.save_toy_npy_traj(opt, fn, xs.detach().cpu().numpy())

    def log_sb_alternate_train(self, opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        print("[{0}] stage {1}/{2} | ep {3}/{4} | train_it {5}/{6} | lr:{7} | loss:{8} | time:{9}"
            .format(
                util.magenta("SB {}".format(direction)),
                util.cyan("{}".format(1+stage)),
                opt.num_stage,
                util.cyan("{}".format(1+ep)),
                num_epoch,
                util.cyan("{}".format(1+it+opt.num_itr*ep)),
                opt.num_itr*num_epoch,
                util.yellow("{:.2e}".format(lr)),
                util.red("{:+.4f}".format(loss.item())),
                util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))

    ''' ==============================================================================
        ====================   Ending module of alternative trainig ==================
        ============================================================================== '''

    def sb_joint_train(self, opt):
        policy_f, policy_b = self.z_f, self.z_b
        policy_f = activate_policy(policy_f)
        policy_b = activate_policy(policy_b)
        optimizer_f, _, sched_f = self.get_optimizer_ema_sched(policy_f)
        optimizer_b, _, sched_b = self.get_optimizer_ema_sched(policy_b)

        ts      = self.ts
        batch_x = opt.samp_bs
        SYNTAX = f'{opt.problem_name}_lr_{opt.lr}_decay_{opt.lr_gamma}_sigma_{opt.sigma_max}_node_{opt.hidden_nodes}_block_{opt.blocks}_batch_{opt.samp_bs}_seed_{opt.seed}'
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
                util.save_toy_npy_traj(opt, f'{SYNTAX}_it{it+1}', xs_b.detach().cpu().numpy())
        with open(f'./results/{SYNTAX}_baseline.npy', 'wb') as f:
            np.save(f, self.p.samples)

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
    #run.sb_joint_train(opt)
    run.sb_alternate_train(opt)

if __name__ == "__main__":
    if not opt.cpu:
        with torch.cuda.device(opt.gpu):
            main(opt)
    else: 
        main(opt)
