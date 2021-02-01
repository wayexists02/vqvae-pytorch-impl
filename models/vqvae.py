import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder
from .quantize import *

from settings import *


class VqVae(nn.Module):

    def __init__(self, feature_dim, n_resid, codebook_size):
        super(VqVae, self).__init__()
            
        self.encoder = Encoder(feature_dim, n_resid)
        self.quantize = Quantize(codebook_size, feature_dim)
        self.decoder = Decoder(feature_dim, n_resid)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, z_q_for_backward = self.quantize(z_e)
        x_rec = self.decoder(z_q)

        return x_rec, z_e, z_q_for_backward


class VqVaeChannelWise(nn.Module):

    def __init__(self, feature_dim, n_resid, codebook_size):
        super(VqVaeChannelWise, self).__init__()
            
        codebook_dim = (1, IMAGE_SIZE//4, IMAGE_SIZE//4)

        self.encoder = Encoder(feature_dim, n_resid)
        self.quantize = QuantizeChannelWise(codebook_size, codebook_dim)
        self.decoder = Decoder(feature_dim, n_resid)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, z_q_for_backward = self.quantize(z_e)
        x_rec = self.decoder(z_q)

        return x_rec, z_e, z_q_for_backward


class VqVaeEMA(nn.Module):

    def __init__(self, feature_dim, n_resid, codebook_size):
        super(VqVaeEMA, self).__init__()
            
        self.encoder = Encoder(feature_dim, n_resid)
        self.quantize = QuantizeEMA(codebook_size, feature_dim)
        self.decoder = Decoder(feature_dim, n_resid)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q = self.quantize(z_e)
        x_rec = self.decoder(z_q)
        
        return x_rec, z_e, z_q


class VqVaeEMAChannelWise(nn.Module):

    def __init__(self, feature_dim, n_resid, codebook_size):
        super(VqVaeEMAChannelWise, self).__init__()
            
        codebook_dim = (1, IMAGE_SIZE//4, IMAGE_SIZE//4)

        self.encoder = Encoder(feature_dim, n_resid)
        self.quantize = QuantizeEMAChannelWise(codebook_size, codebook_dim)
        self.decoder = Decoder(feature_dim, n_resid)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q = self.quantize(z_e)
        x_rec = self.decoder(z_q)
        
        return x_rec, z_e, z_q
