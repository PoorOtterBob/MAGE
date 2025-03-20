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
        self.encoder1 = SwiGLU_FFN(self.seq_len, dim)
        self.encoder2 = SwiGLU_FFN(self.seq_len*(self.input_dim-1), dim)
        self.encoder3 = SwiGLU_FFN(self.seq_len*(self.input_dim-1), dim)

        # self.backcast = SwiGLU_FFN(dim, dim)

        self.decoder1 = SwiGLU_FFN(dim, self.output_dim*self.horizon)
        self.decoder2 = SwiGLU_FFN(dim, self.output_dim*self.horizon)

        self.position1 = nn.Parameter(torch.zeros((self.node_num, dim)))
        self.position2 = nn.Parameter(torch.zeros((self.node_num, dim)))
        self.position3 = nn.Parameter(torch.zeros((self.node_num, dim)))

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        # self.norm = BatchNorm(dim)

        self.module1 = Parallel_Graphormer(dim, head, rank, self.node_num)
        self.module2 = Parallel_Graphormer(dim, head, rank, self.node_num)
        # self.module = Serial_Graphormer(dim, head, rank, self.node_num)

        self.alpha = nn.Parameter(torch.tensor(0.))
        self.beta = nn.Parameter(torch.tensor(0.))
        print('bingxing')

    '''def forward(self, x, label=None): 
        z = x[..., 1:].transpose(1, 2).reshape(-1, 1, self.node_num, self.seq_len*(self.input_dim-1))
        z = self.encoder2(z) + self.norm2(self.position2)

        label = label.transpose(1, 2).reshape(-1, 1, self.node_num, self.seq_len*(self.input_dim-1))
        label = self.encoder3(label) +  self.norm3(self.position3)

        x = x[..., 0:1].transpose(1, -1)
        x = self.encoder1(x) +  self.norm1(self.position1)

        # x = x.transpose(1, 2).reshape(-1, 1, self.node_num, self.seq_len*self.input_dim)
        # x = self.encoder3(x) +  self.norm(self.position1)

        # x = self.module1(x, z) # + x
        # x = self.module2(x, label) # + x
        # x = F.sigmoid(self.alpha)*self.module1(x, z) + F.sigmoid(self.beta)*self.module2(x, label)
        # y1 = self.module1(x, z)
        # y2 = self.module2(self.backcast(x) - y1, label)
        # y2 = self.module2(x - y1, label)

        # y1 = self.decoder1(y1).transpose(1, -1)
        # y2 = self.decoder2(y2).transpose(1, -1)
        # x = self.decoder1(x).transpose(1, -1)
        # return y1 + y2
        # return x.transpose(1, -1)

        x = self.decoder1(self.module1(x, z)) + self.decoder2(self.module2(x, label))
        return x.transpose(1, -1)'''
    
    def forward(self, x, label=None): 
        z = x[..., 1:].transpose(1, 2).reshape(-1, 1, self.node_num, self.seq_len*(self.input_dim-1))
        z = self.encoder2(z) + self.norm2(self.position2)

        label = label.transpose(1, 2).reshape(-1, 1, self.node_num, self.seq_len*(self.input_dim-1))
        label = self.encoder3(label) +  self.norm3(self.position3)

        x = x[..., 0:1].transpose(1, -1)
        x = self.encoder1(x) +  self.norm1(self.position1)

        # x = x.transpose(1, 2).reshape(-1, 1, self.node_num, self.seq_len*self.input_dim)
        # x = self.encoder3(x) +  self.norm(self.position1)

        # x = self.module1(x, z)
        # x = self.module2(x, label)
        x = F.sigmoid(self.alpha)*self.module1(x, z) + F.sigmoid(self.beta)*self.module2(x, label)
        x = self.decoder1(x)
        return x.transpose(1, -1)
    
class Parallel_Graphormer(nn.Module):
    def __init__(self, dim, head=4, rank=None, node_num=None):
        super(Parallel_Graphormer, self).__init__()
        # self.MHA = Light_Spattention(dim, head, node_num)
        self.MHA = Lowrank_Spattention(dim, dim, rank, head)
        self.FFN = SwiGLU_FFN(dim, dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        # self.norm = BatchNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(math.log(9)))
        self.beta = nn.Parameter(torch.tensor(math.log(math.sqrt(2) + 1)))

        self.gamma = nn.Parameter(torch.tensor(math.log(9)))
        self.delta = nn.Parameter(torch.tensor(math.log(math.sqrt(2) + 1)))
    def forward(self, x, z):
        x = self.norm1(x) # pre-norm
        z = self.norm2(F.sigmoid(self.gamma)*x + F.sigmoid(self.delta)*z) # pre-norm
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
        self.Q = nn.Parameter(torch.zeros((dim, dim)))
        self.K = nn.Parameter(torch.randn((dim, dim)))

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
        
    def forward(self, x):
        b, t, n, f = x.shape
        # x = x.reshape(-1, -1, -1, self.head, self.head_dim)
        # (b, t, n, f) -> (b, t, n, h, d), d = f/h
        q = (x @ self.Q).reshape(b, t, n, self.head, self.head_dim)
        k = (x @ self.K).reshape(b, t, n, self.head, self.head_dim)
        # q, k = q / (torch.norm(q, p=2)+1e-5), k / torch.norm(k, p=2)
        
        # scale = torch.einsum('btnhd, n -> bthd', k, self.One) 
        # scale = torch.einsum('bthd, btnhd -> btnh', scale, q).unsqueeze(-1) / n + 1

        x = x.reshape(b, t, n, self.head, self.head_dim)

        attn = torch.einsum('btmhd, btmhs -> btshd', x, k)
        attn = torch.einsum('btshd, btnhs -> btnhd', attn, q) / n
        # attn = attn * scale**(-1)
  
        x = F.sigmoid(self.alpha)*x + F.sigmoid(self.beta)*attn
        return x.reshape(b, t, n, -1)


class SwiGLU_FFN(nn.Module):
    def __init__(self, dim_in, dim_out, expand_ratio=4, dropout=0.3):
        super(SwiGLU_FFN, self).__init__()
        self.W1 = nn.Linear(dim_in, expand_ratio*dim_in)
        self.W2 = nn.Linear(dim_in, expand_ratio*dim_in)
        self.W3 = nn.Linear(expand_ratio*dim_in, dim_out)
        self.dropout =  nn.Dropout(dropout)

    def forward(self, x):
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))
    

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


    def forward(self, x, z): 
        b, t, n, f = x.shape
        q = self.query(z).reshape(b, t, n, self.head, self.head_dim)
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
        return self.norm(x.transpose(1, -1)).transpose(1, -1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x)