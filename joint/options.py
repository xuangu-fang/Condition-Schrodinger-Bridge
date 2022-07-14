import numpy as np
import os
import argparse
import random
import torch

import util

from ipdb import set_trace as debug

import ml_collections

def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.seed = 999
    config.T = 1.0
    config.interval = 100
    config.train_method = 'joint'
    config.t0 = 0
    #config.problem_name = 'gmm'
    config.num_itr = 2000
    config.eval_itr = 500
    config.forward_net = 'toy'
    config.backward_net = 'toy'

    # sampling
    #config.samp_bs = 1000 # cantor server doesn't support large batch size
    config.samp_bs = 500
    config.sigma_min = 0.01
    config.sigma_max = 0.3

    # optimization
    # config.optim = optim = ml_collections.ConfigDict()
    config.weight_decay = 0
    config.optimizer = 'AdamW'
    config.lr = 4e-3
    config.lr_gamma = 0.8

    # network structure
    config.hidden_nodes = 128
    config.blocks = 1

    model_configs=None
    return config, model_configs

def set():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-name",   type=str,   default='Scurve')
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--gpu",            type=int,   default=0,        help="GPU device")
    parser.add_argument("--load",           type=str,   default=None,     help="load the checkpoints")
    parser.add_argument("--dir",            type=str,   default=None,     help="directory name to save the experiments under results/")
    parser.add_argument("--group",          type=str,   default='0',      help="father node of directionary for saving checkpoint")
    parser.add_argument("--name",           type=str,   default='debug',  help="son node of directionary for saving checkpoint")
    parser.add_argument("--log-fn",         type=str,   default=None,     help="name of tensorboard logging")
    parser.add_argument("--log-tb",         action="store_true",          help="logging with tensorboard")
    parser.add_argument("--cpu",            action="store_true",          help="use cpu device")

    # --------------- SB model ---------------
    parser.add_argument("--t0",             type=float, default=1e-2,     help="time integral start time")
    parser.add_argument("--T",              type=float, default=1.,       help="time integral end time")
    parser.add_argument("--interval",       type=int,   default=100,      help="number of interval")
    parser.add_argument("--forward-net",    type=str,   default='toy', choices=['toy','Unet','ncsnpp'], help="model class of forward nonlinear drift")
    parser.add_argument("--backward-net",   type=str,   default='toy', choices=['toy','Unet','ncsnpp'], help="model class of backward nonlinear drift")
    parser.add_argument("--sde-type",       type=str,   default='ve', choices=['ve', 'vp', 'simple'])
    parser.add_argument("--sigma-max",      type=float, default=50,       help="max diffusion for VESDE")
    parser.add_argument("--sigma-min",      type=float, default=0.01,     help="min diffusion for VESDE")

    parser.add_argument("--hidden_nodes",   type=int, default=128,        help="hidden nodes")
    parser.add_argument("--blocks",         type=int, default=2,          help="NN layers")


    # --------------- SB training & sampling (corrector) ---------------
    parser.add_argument("--train-method",   type=str, default=None,       help="algorithm for training SB" )
    parser.add_argument("--eval-itr",       type=int, default=200,        help="[sb joint train] frequency of evaluation")
    parser.add_argument("--samp_bs",        type=int,                     help="[sb train] batch size for all trajectory sampling purposes")
    parser.add_argument("--num_itr",        type=int,                     help="[sb train] number of training iterations (for each epoch)")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--lr",             type=float,                   help="learning rate")
    parser.add_argument("--lr-f",           type=float, default=None,     help="learning rate for forward network")
    parser.add_argument("--lr-b",           type=float, default=None,     help="learning rate for backward network")
    parser.add_argument("--lr_gamma",       type=float, default=1.0,      help="learning rate decay ratio")
    parser.add_argument("--lr_step",        type=int,   default=500,     help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0,      help="weight decay rate")
    parser.add_argument("--optimizer",      type=str,   default='AdamW',  help="optmizer")
    parser.add_argument("--grad-clip",      type=float, default=None,     help="clip the gradient")
    parser.add_argument("--noise-type",     type=str,   default='gaussian', choices=['gaussian','rademacher'], help='choose noise type to approximate Trace term')

    problem_name = parser.parse_args().problem_name
    default_config, model_configs = {
        'toy':          get_default_configs,
    }.get('toy')()
    parser.set_defaults(**default_config)

    opt = parser.parse_args()

    # ========= seed & torch setup =========
    if opt.seed is not None:
        # https://github.com/pytorch/pytorch/issues/7068
        seed = opt.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.autograd.set_detect_anomaly(True)
    
    # ========= auto setup & path handle =========
    opt.device='cuda:'+str(opt.gpu)
    opt.model_configs = model_configs
    if opt.lr is not None:
        opt.lr_f, opt.lr_b = opt.lr, opt.lr

    # ========= print options =========
    for o in vars(opt):
        print(util.green(o),":",util.yellow(getattr(opt,o)))
    print()

    return opt
