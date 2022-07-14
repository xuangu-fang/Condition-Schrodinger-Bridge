import torch
import torch.nn as nn
from models.utils import *


class ToyPolicy(torch.nn.Module):
    def __init__(self, data_dim=2, hidden_dim=256, blocks=3, zero_out_last_layer=False):
        super(ToyPolicy,self).__init__()

        self.time_embed_dim = hidden_dim // 2
        self.zero_out_last_layer = zero_out_last_layer
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )

        self.x_module = ResNet_FC(data_dim, hidden_dim, num_res_blocks=blocks)

        self.out_module = nn.Sequential(
            nn.Linear(hid,hid),
            SiLU(),
            nn.Linear(hid, data_dim),
        )
        if zero_out_last_layer:
            self.out_module[-1] = zero_module(self.out_module[-1])

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype


    def forward(self,x, t):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """

        # make sure t.shape = [T]
        if len(t.shape)==0:
            t=t[None]

        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)

        out = self.out_module(x_out + t_out)


        return out


class ToyPolicy_condi(ToyPolicy):
    def __init__(self, data_dim=2, hidden_dim=256, blocks=3, zero_out_last_layer=False):
        super(ToyPolicy_condi,self).__init__( data_dim, hidden_dim, blocks, zero_out_last_layer)

        # fang: add extra subnet to handle x_condition
        self.x_condi_module = ResNet_FC(data_dim, hidden_dim, num_res_blocks=blocks)

        # should have same shape of x, set at when sample from p
        self.x_condi = None

    def set_x_condi(self,x_condi):
        self.x_condi = x_condi

    def forward(self,x, t):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """

        # make sure t.shape = [T]
        if len(t.shape)==0:
            t=t[None]

        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)

        x_out_condi = self.x_condi_module(self.x_condi)

        out = self.out_module(x_out + t_out + x_out_condi)

        return out