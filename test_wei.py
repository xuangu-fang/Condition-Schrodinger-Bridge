#!/usr/bin/env python3
import os, sys

import argparse

parser = argparse.ArgumentParser(description='Grid search')
parser.add_argument('-id', default=1, type=int, help='problem id')
parser.add_argument('-sn', default=10, type=int, help='Sampling Epochs')
parser.add_argument('-lr', default=2e-3, type=float, help='learning rate')
parser.add_argument('-lr_gamma', default=0.8, type=float, help='learning rate')
parser.add_argument('-node', default=128, type=int, help='node')
parser.add_argument('-layer', default=1, type=int, help='layer')
parser.add_argument('-batch', default=500, type=int, help='batch size')
parser.add_argument('-gpu', default=0, type=int, help='gpu id')


pars = parser.parse_args()


syntax = f'--num_itr {pars.sn} --lr {pars.lr} --lr_gamma {pars.lr_gamma} --hidden_nodes {pars.node} --blocks {pars.layer} --samp_bs {pars.batch} --gpu {pars.gpu} --lr {pars.lr} '

if pars.id == 1:
    os.system('python runner.py --problem-name Scurve --forward-net toy --backward-net toy  --dir ./ ' + syntax)
elif pars.id == 2:
    os.system('python runner.py --problem-name Spiral --forward-net toy --backward-net toy  --dir ./ ' + syntax)
elif pars.id == 3:
    os.system('python runner.py --problem-name Circle --forward-net toy --backward-net toy  --dir ./ ' + syntax)
elif pars.id == 4:
    os.system('python runner.py --problem-name Moon --forward-net toy --backward-net toy  --dir ./ ' + syntax)


#python runner.py --problem-name Spiral --forward-net toy --backward-net toy  --dir ./
#python runner.py --problem-name Moon --forward-net toy --backward-net toy  --dir ./ --gpu 1
#python runner.py --problem-name Circle --forward-net toy --backward-net toy  --dir ./ --gpu 1
#python runner.py --problem-name Pinwheel --forward-net toy --backward-net toy  --dir ./
