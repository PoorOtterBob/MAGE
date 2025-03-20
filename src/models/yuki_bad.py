import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from src.base.model import BaseModel
import sys

class STE(BaseModel):
    def __init__(self, dim, rank, head=4, **args):
        super(STE, self).__init__(**args)
        self.encoder_indicate = SwiGLU_FFN(self.seq_len, dim)
        self.encoder_feature = SwiGLU_FFN(self.seq_len*(self.input_dim - 1), dim)
        self.decoder = SwiGLU_FFN(dim, self.output_dim*self.horizon)
        self.position = nn.Parameter(torch.zeros((self.seq_len, self.node_num, dim)))
        self.norm = RMSNorm(dim)
        # self.norm = BatchNorm(dim)

        self.module = Parallel_Graphormer(dim, head, rank, self.node_num)
        # self.module = Serial_Graphormer(dim, head, rank, self.node_num)

    def forward(self, x, label=None): 
        b, t, n, f = x.shape
        indicate = self.encoder_indicate(x[..., 0:1])
        feature = self.encoder_feature(x[..., 1:].transpose(-1, -2).reshape(b, -1, n, 1))
        output = self.module(indicate, feature)
        output = self.decoder(output)
        return output


class Parallel_Graphormer(nn.Module):
    def __init__(self, dim, head=4, rank=None, node_num=None):
        super(Parallel_Graphormer, self).__init__()
        self.MHA = Light_Spattention(dim, head, node_num)
        # self.MHA = Lowrank_Spattention(dim, dim,gpustat rank, head)
        self.FFN = SwiGLU_FFN(dim, dim)
        self.norm = RMSNorm(dim)
        # self.norm = BatchNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(math.log(9)))
        self.beta = nn.Parameter(torch.tensor(math.log(math.sqrt(2) + 1)))

    def forward(self, x, z):
        x = self.norm(x) # pre-norm
        return F.sigmoid(self.alpha)*self.FFN(x) + F.sigmoid(self.beta)*(self.MHA(x, z))
    
class Serial_Graphormer(nn.Module):
    def __init__(self, dim, head=4, rank=None, node_num=None):
        super(Serial_Graphormer, self).__init__()
        # self.MHA = Light_Spattention(dim, head, node_num)
        self.MHA = Lowrank_Spattention(dim, dim, rank, head)
        self.FFN = SwiGLU_FFN(dim, dim)
        self.norm = RMSNorm(dim)
        # self.norm = BatchNorm(dim)

    def forward(self, x):
        x = self.MHA(self.norm(x)) + x # pre-norm
        x = self.FFN(self.norm(x)) + x
        return x



class Light_Spattention(nn.Module):
    def __init__(self, dim, head=4, node_num=None):
        super(Light_Spattention, self).__init__()
        # self.Q = nn.Parameter(torch.zeros((dim, dim)))
        # self.K = nn.Parameter(torch.randn((dim, dim)))
        self.Q = nn.Conv2d(dim, dim, kernel_size=1)
        self.K = nn.Conv2d(dim, dim, kernel_size=1)
        self.reset_parameter()

        try:
            # self.One = nn.Parameter(torch.ones(node_num, requires_grad=False))
            self.One = nn.Parameter(torch.ones((node_num), requires_grad=False))
            print('We have node_num as a hyper-parameter')
        except:
            self.node_num = node_num
            print('We do not have node_num as a hyper-parameter')

        # self.alpha = nn.Parameter(torch.ones(head, 1))
        # self.beta = nn.Parameter(torch.ones(head, 1))
        # self.gamma = nn.Parameter(torch.ones(head, 1))

        self.alpha = nn.Parameter(torch.tensor([math.log(9) for _ in range(head)]).unsqueeze(-1))
        self.beta = nn.Parameter(torch.tensor([math.log(9) for _ in range(head)]).unsqueeze(-1))
        self.gamma = nn.Parameter(torch.tensor([math.log(9) for _ in range(head)]).unsqueeze(-1))

        self.dim = dim
        self.head = head
        self.head_dim = self.dim // self.head

    def reset_parameter(self):
        for param in self.Q.parameters():
            if len(param.shape) > 1:  # 找到线性层的参数
                param.data.fill_(0)
        print('reset parameter')

    def forward(self, x, z):
        b, f, n, _ = x.shape

        q = self.Q(z).reshape(b, self.head_dim, self.head, n)
        k = self.K(x).reshape(b, self.head_dim, self.head, n)
        x = x.reshape(b, self.head_dim, self.head, n)

        attn = torch.einsum('bdhn, bshn -> bdhs', x, k)
        attn = torch.einsum('bdhs, bshn -> btnhd', attn, q) / n
  
        x = F.sigmoid(self.alpha)*x + F.sigmoid(self.beta)*attn
        return x.reshape(b, f, n, 1)


class SwiGLU_FFN(nn.Module):
    def __init__(self, dim_in, dim_out, expand_ratio=4, dropout=0.3):
        super(SwiGLU_FFN, self).__init__()
        # self.W1 = nn.Linear(dim_in, expand_ratio*dim_in)
        # self.W2 = nn.Linear(dim_in, expand_ratio*dim_in)
        # self.W3 = nn.Linear(expand_ratio*dim_in, dim_out)
        self.dropout =  nn.Dropout(dropout)
        
        self.W1 = nn.Conv2d(dim_in, expand_ratio*dim_out, kernel_size=1)
        self.W2 = nn.Conv2d(dim_in, expand_ratio*dim_out, kernel_size=1)
        self.W3 = nn.Conv2d(expand_ratio*dim_out, dim_out, kernel_size=1)
    def forward(self, x):
        # x = x.transpose(1, -1)
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))# .transpose(1, -1)
    

class Lowrank_Spattention(nn.Module):
    def __init__(self, dim, dim_attn, rank, head=4):
        super(Lowrank_Spattention, self).__init__()
        if rank == 0:
            raise 
        else:
            print('rank number is', rank)
        if head == 0:
            raise 
        else:
            print('head number is', head)

        self.head_dim = dim_attn // head
        
        
        self.query = nn.Linear(dim, dim_attn)
        self.key = nn.Parameter(torch.randn((rank, head, self.head_dim)))
        self.value = nn.Linear(dim, dim)

        self.alpha = nn.Parameter(torch.tensor([math.log(9) for _ in range(head)]).unsqueeze(-1))
        self.beta = nn.Parameter(torch.tensor([math.log(0.01) for _ in range(head)]).unsqueeze(-1))

        self.rank = rank
        self.head = head


    def forward(self, x): 
        b, t, n, f = x.shape
        q = self.query(x).reshape(b, t, n, self.head, self.head_dim)
        attn = torch.einsum('btnhd, rhd -> btnhr', q, self.key) / self.head_dim**0.5
        x = self.value(x).reshape(b, t, n, self.head, self.head_dim)
        v = torch.einsum('btnhr, btnhd -> btrhd', F.softmax(attn, dim=-1), x)
        v = torch.einsum('btnhr, btrhd -> btnhd', F.softmax(attn, dim=-3), v)
        v = F.sigmoid(self.alpha)*x + F.sigmoid(self.beta)*v
        return v.reshape(b, t, n, f)
    

class BatchNorm(nn.Module):
    def __init__(self, dim):
        super(BatchNorm, self).__init__()
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.norm(x)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # variance = x.pow(2).mean(-1, keepdim=True)
        variance = x.pow(2).mean(1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x)