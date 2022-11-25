import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence

from itertools import chain
from scipy.stats import truncnorm
import math
import numpy as np


## FUNCTION BLOCKS ##
class ConvBlock(nn.Module):
    def __init__(self, 
                 in_dim=None, 
                 out_dim=None, 
                 kernel=None, 
                 stride=1, 
                 padding=0, 
                 dilation=1,
                 batchnorm=True,
                 dropout=None,
                 nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvBlock, self).__init__()

        modules =  nn.ModuleList([
            nn.ReplicationPad1d(padding),
            nn.Conv1d(in_dim, out_dim, kernel, stride, 0, dilation)])

        if batchnorm is True:
            modules.append(nn.BatchNorm1d(out_dim))
        if nonlinearity is not None:
            modules.append(nonlinearity)
        if dropout is not None:
            modules.append(nn.Dropout(p=dropout))
            
        self.layer = nn.Sequential(*modules)

    def forward(self, x):
        return self.layer(x) 


class MaskedConvBlock(nn.Module):
    def __init__(self, 
                 in_dim=None, 
                 out_dim=None, 
                 kernel=None, 
                 stride=1, 
                 padding=0, 
                 dilation=1,
                 batchnorm=True,
                 dropout=None,
                 nonlinearity=nn.LeakyReLU(0.2)):
        super(MaskedConvBlock, self).__init__()

        modules1 = nn.ModuleList([
            nn.ReplicationPad1d(padding),
            nn.Conv1d(in_dim, out_dim, kernel, stride, 0, dilation)])
        modules2 = nn.ModuleList([])

        if batchnorm is True:
            modules2.append(nn.BatchNorm1d(out_dim))
        if nonlinearity is not None:
            modules2.append(nonlinearity)
        if dropout is not None:
            modules2.append(nn.Dropout(p=dropout))
            
        self.layer1 = nn.Sequential(*modules1)
        self.layer2 = nn.Sequential(*modules2)

    def forward(self, x, mask=None, train=True):
        x = self.layer1(x)
        if train is True: 
            x = mask(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = x
        x = self.layer2(x)
        return x


class CausalConvBlock(nn.Module):
    def __init__(self, 
                 in_dim=None, 
                 out_dim=None, 
                 kernel=None, 
                 stride=1,
                 dilation=1,
                 batchnorm=True,
                 nonlinearity=nn.LeakyReLU(0.2),
                 dropout=None):
        super(CausalConvBlock, self).__init__()

        self.left_padding = dilation * (kernel - 1)
        padding = 0
        
        modules = [nn.Conv1d(in_dim, out_dim, kernel, stride, padding, dilation)]

        if batchnorm is True:
            modules.append(nn.BatchNorm1d(out_dim))
        if nonlinearity is not None:
            modules.append(nonlinearity)
        if dropout is not None:
            modules.append(nn.Dropout(p=dropout))
            
        self.layer = nn.Sequential(*modules)

    def forward(self, x):
        '''
        https://github.com/pytorch/pytorch/issues/1333
        '''
        x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0), mode="replicate").squeeze(2)
        return self.layer(x)


class MaskedCausalConvBlock(nn.Module):
    def __init__(self, 
                 in_dim=None, 
                 out_dim=None, 
                 kernel=None, 
                 stride=1,
                 dilation=1,
                 batchnorm=True,
                 nonlinearity=nn.LeakyReLU(0.2),
                 dropout=None):
        super(MaskedCausalConvBlock, self).__init__()

        self.left_padding = dilation * (kernel - 1)
        padding = 0
        
        self.conv = nn.Conv1d(in_dim, out_dim, kernel, stride, padding, dilation)
        modules = nn.ModuleList([])

        if batchnorm is True:
            modules.append(nn.BatchNorm1d(out_dim))
        if nonlinearity is not None:
            modules.append(nonlinearity)
        if dropout is not None:
            modules.append(nn.Dropout(p=dropout))
            
        self.layer = nn.Sequential(*modules)

    def forward(self, x, mask=None, train=True):
        '''
        https://github.com/pytorch/pytorch/issues/1333
        '''
        x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0), mode="replicate").squeeze(2)
        x = self.conv(x)
        if train is True: 
            x = mask(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = x        
        x = self.layer(x)
        return x


class FCBlock(nn.Module):
    def __init__(self,
                 in_dim=None,  
                 out_dim=None,
                 batchnorm=True,
                 nonlinearity=nn.LeakyReLU(0.2),
                 dropout=None):
        super(FCBlock, self).__init__()
        self.batchnorm = batchnorm

        self.first_layer = nn.Linear(in_dim, out_dim)
        self.second_layer = nn.BatchNorm1d(out_dim)

        modules = list()
        if nonlinearity is not None:
            modules.append(nonlinearity)
        if dropout is not None:
            modules.append(nn.Dropout(p=dropout))
        self.post_layer = nn.Sequential(*modules)

    def forward(self, x):
        first_out = self.first_layer(x)
        if self.batchnorm is True:
            second_out = self.second_layer(first_out.transpose(1, 2))
            out = second_out.transpose(1, 2)
        elif self.batchnorm is False:
            out = first_out  
        return self.post_layer(out)


class FCBlock_LN(nn.Module):
    def __init__(self,
                 in_dim=None,  
                 out_dim=None,
                 nonlinearity=nn.LeakyReLU(0.2),
                 dropout=None):
        super(FCBlock_LN, self).__init__()

        self.first_layer = nn.Linear(in_dim, out_dim)
        self.second_layer = nn.LayerNorm(out_dim)

        modules = list()
        if nonlinearity is not None:
            modules.append(nonlinearity)
        if dropout is not None:
            modules.append(nn.Dropout(p=dropout))
        self.post_layer = nn.Sequential(*modules)

    def forward(self, x):
        first_out = self.first_layer(x)
        # layer norm
        second_out = self.second_layer(first_out)

        return self.post_layer(second_out)


class MaskedFCBlock(nn.Module):
    def __init__(self,
                 in_dim=None,  
                 out_dim=None,
                 batchnorm=True,
                 nonlinearity=nn.LeakyReLU(0.2),
                 dropout=None):
        super(MaskedFCBlock, self).__init__()
        self.batchnorm = batchnorm

        self.first_layer = nn.Linear(in_dim, out_dim)
        self.second_layer = nn.BatchNorm1d(out_dim)

        modules = list()
        if nonlinearity is not None:
            modules.append(nonlinearity)
        if dropout is not None:
            modules.append(nn.Dropout(p=dropout))
        self.post_layer = nn.Sequential(*modules)

    def forward(self, x, mask=None, train=True):
        first_out = self.first_layer(x)
        if train is True: 
            first_out = mask(first_out)
        else:
            first_out = first_out   
        if self.batchnorm is True:
            second_out = self.second_layer(first_out.transpose(1, 2))
            out = second_out.transpose(1, 2)
        elif self.batchnorm is False:
            out = first_out  
        return self.post_layer(out)


class MaskedFCBlock_LN(nn.Module):
    def __init__(self,
                 in_dim=None,  
                 out_dim=None,
                 nonlinearity=nn.LeakyReLU(0.2),
                 dropout=None):
        super(MaskedFCBlock_LN, self).__init__()

        self.first_layer = nn.Linear(in_dim, out_dim)
        self.second_layer = nn.LayerNorm(out_dim)

        modules = list()
        if nonlinearity is not None:
            modules.append(nonlinearity)
        if dropout is not None:
            modules.append(nn.Dropout(p=dropout))
        self.post_layer = nn.Sequential(*modules)

    def forward(self, x, mask=None, train=True):
        first_out = self.first_layer(x)
        if train is True: 
            first_out = mask(first_out)
        else:
            first_out = first_out   
        # layernorm
        second_out = self.second_layer(first_out)

        return self.post_layer(second_out)


class TruncatedNorm(nn.Module):
    def __init__(self):
        super(TruncatedNorm, self).__init__()

    def forward(self, size, threshold=2.):
        values = truncnorm.rvs(-threshold, threshold, size=size)
        return values.astype('float32')


class SplitScore(nn.Module):
    def __init__(self, dim=-1):
        super(SplitScore, self).__init__()

    def forward(self, x):
        '''
        0~11: relative ioi
        11~22: relative duration 
        22~110: pitch 
        110~112: is same onset 
        112~114: is top voice
        114~125: stacked note # in group
        125~136: note position within group
        136~138: staff 
        138~140: is downbeat
        '''
        x = torch.cat([x[:,:,:110], x[:,:,112:]], axis=-1)
        return x


class Mask(nn.Module):
    def __init__(self, m):
        super(Mask, self).__init__()    
        self.note = torch.sign(
            torch.abs(torch.sum(m, dim=-1)))
        self.chord = torch.sign(
            torch.abs(torch.sum(m.transpose(1, 2), dim=-1)))

    def forward(self, x):
        n, t, d = x.size(0), x.size(1), x.size(2)
        if t == self.note.size(1):
            mask = self.note 
        elif t == self.chord.size(1):
            mask = self.chord
        mask_expand = mask.unsqueeze(-1).expand(n, t, d)
        out = torch.mul(x, mask_expand)  
        return out   


class Note2Group(nn.Module):
    def __init__(self):
        super(Note2Group, self).__init__()    

    def forward(self, x, m):
        out = torch.matmul(x.transpose(1, 2), m).transpose(1, 2)
        m_ = torch.empty_like(m).copy_(m)
        m_sum = torch.sum(m_, dim=1).unsqueeze(-1)
        m_sum = torch.where(m_sum==0, torch.ones_like(m_sum), m_sum)
        out = torch.div(out, m_sum)
        return out
    
    def reverse(self, x, m):
        out = torch.matmul(
            x.transpose(1, 2), m.transpose(1, 2)).transpose(1, 2)
        return out


class Reparameterize(nn.Module):
    def __init__(self, device=None):
        super(Reparameterize, self).__init__()    
        self.trunc = TruncatedNorm()
        self.device = device

    def forward(self, mu, logvar, trunc=False, threshold=None):
        epsilon = torch.randn_like(logvar)
        if trunc is True:
            epsilon = self.trunc(
                logvar.shape, threshold=threshold)
            epsilon = torch.from_numpy(epsilon).to(self.device)
        z = mu + torch.exp(0.5 * logvar) * epsilon 
        return z


class StyleEstimator(nn.Module):
    def __init__(self):
        super(StyleEstimator, self).__init__() 
        self.note2group = Note2Group()

    def forward(self, y, clab):
        y_ = torch.empty_like(y).copy_(y)
        clab_ = torch.empty_like(clab).copy_(clab)
        zlab_ = torch.sign(y_ - clab_)
        return zlab_



