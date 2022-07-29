'''
modify from the diff_model.py in CSDI project
Transformer based NN to to handle multi-var time series imputation
The diff_CSDI class is just the NN-based policy in CSB
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from models.utils import *

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu",batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer




class Trans_1d(nn.Module):
    def __init__(self,opt, inputdim=1):
        super().__init__()
        # self.channels = config["channels"]
        self.channels = opt.channels
        self.device = opt.device
        self.opt = opt

        hid = opt.hidden_nodes

        self.time_embed_dim = hid // 2

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, opt.data_dim[0]),
        )

        
        self.pos_emb = self._build_embedding(opt.data_dim[0],self.time_embed_dim//2).unsqueeze(0) # (1,L,time_embed)

        self.pos_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, self.channels),
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        # self.position_projection = Conv1d_with_init(inputdim, self.channels, 1)

        # self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        # nn.init.zeros_(self.output_projection2.weight)

        self.transforms_layer = get_torch_trans(heads=opt.nheads, layers=1, channels=opt.channels)
        


        self.out_module = nn.Sequential(
                    nn.Linear(opt.data_dim[0],hid),
                    SiLU(),
                nn.Linear(hid, opt.data_dim[0]),
                        )


        # fang: set the condi_info as the class attributes
        self.x_condi = None

        # fang: set the side_info as the class attributes
        self.side_info = None

    def set_x_condi(self,x_condi):
        self.x_condi = x_condi

    def set_side_info(self, observed_tp, cond_mask):
        # self.side_info = self.get_side_info( observed_tp, cond_mask)
        pass

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table.to(self.device)

    def forward(self, x, t):
        # make sure t.shape = [T]
        if len(t.shape)==0:
            t=t[None]
        
        B = x.shape[0]

        pos_emb = self.pos_emb.repeat(B,1,1)# (B,L,pos_emb)
        pos_emb = self.pos_module(pos_emb)# (B,L,C)

        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb) # (B,L)
        t_out = t_out.unsqueeze(1).repeat(1,self.channels,1) # (B,C,L)

        
        x = self.input_projection(x.unsqueeze(1)) # (B,L)->(B,1,L)->(B,C,L)

        x = x.permute(0,2,1)# (B,C,L)-># (B,L,C)

        x = self.transforms_layer(x + pos_emb) # (B,L,C)->(B,L,C)

        x = x.permute(0,2,1)  # (B,L,C)->(B,C,L)

        x = x + t_out# (B,L,C)

        x = self.output_projection2(x).squeeze()# (B,C,L)->(B,1,L)->(B,L)
        
        # out = self.out_module(x+ t_out)
        out = x

        return out
