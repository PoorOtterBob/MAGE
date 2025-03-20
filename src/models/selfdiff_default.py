import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.base.model import BaseModel

import sys

# Mixture of Adptive Graph Experts
class MAGE(BaseModel):
    def __init__(self, model_args, **args):
        super(MAGE, self).__init__(**args)
        self.args = model_args
        self.encoder = nn.Linear(self.seq_len*self.input_dim, self.args.model_dim)
        self.decoder = nn.Linear(self.args.model_dim, self.horizon*self.output_dim)
        self.decoder2 = nn.Linear(self.args.model_dim, self.horizon*self.output_dim)

        self.prompt = Prompt_Pool(self.args)
        self.backbone = nn.ModuleList([
            Pre_Norm_Transformer(self.args.model_dim, self.args.recur_num, self.node_num, self.args.blocknum)
            for _ in range(self.args.blocknum)
        ])
        # self.backbone1 = Pre_Norm_Transformer(self.args.model_dim, self.args.recur_num, self.node_num, blocknum=self.args.blocknum)
        # self.backbone2 = Pre_Norm_Transformer(self.args.model_dim, self.args.recur_num, self.node_num, blocknum=self.args.blocknum)
        # self.ffn = Pre_Norm_Transformer(self.args.model_dim, self.args.recur_num, self.node_num, blocknum=self.args.blocknum)
        # self.ffn = SwiGLU_FFN(self.args.model_dim, self.args.model_dim)

    def forward(self, x, label=None):
        x = self.encoder(self.channel_compress(x)) + self.prompt(x)
        for module in self.backbone:
            x = module(x)
        x = self.decoder(x)
        return self.channel_decompress(x)
    
    def channel_compress(self, x):
        return x.transpose(1, 2).reshape(self.args.bs, self.node_num, -1)
    
    def channel_decompress(self, x):
        return x.reshape(self.args.bs, self.node_num, self.horizon, -1).transpose(1, 2)


class Pre_Norm_Transformer(nn.Module):
    def __init__(self, dim, expert_num, node_num, blocknum):
        super(Pre_Norm_Transformer, self).__init__()
        # self.attn = MAGE_Block(dim, expert_num, node_num, depth=blocknum, model_dim=dim)
        self.attn = SelfDiff_Attention(dim, expert_num, node_num, depth=blocknum, model_dim=dim)
        self.attn_norm = RMSNorm(dim)

        self.ffn = SwiGLU_FFN(dim, dim)
        self.ffn_norm = RMSNorm(dim)


    def forward(self, x):
        x = self.attn(self.attn_norm(x)) + x
        x = self.ffn(self.ffn_norm(x)) + x
        return x

class SelfDiff_Attention(nn.Module):
    def __init__(self, dim, expert_num, node_num, depth, model_dim, graph_gen_dim=32, head=8):
        super(SelfDiff_Attention, self).__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.mh = nn.Linear(dim, dim, bias=False)
        self.head = head
        self.head_dim = dim // head
        '''# Differential
        self.E1 = nn.Parameter(torch.rand(2, expert_num, node_num, graph_gen_dim))
        self.E2 = nn.Parameter(torch.rand(2, expert_num, graph_gen_dim, node_num))
        self.router = nn.Sequential(
            nn.Linear(dim, expert_num),
            nn.Softmax(dim=-1)
        )
        
        self.lambda_init = self.lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.coefficient = nn.Parameter(torch.ones(1), requires_grad=False)

    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

    def cal_lambda(self):
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1, keepdim=True).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1, keepdim=True).float())
        lambda_full = - (lambda_1 - lambda_2 + self.lambda_init)
        return lambda_full 

    def cal_coefficient(self):
        return torch.concat([self.coefficient, self.cal_lambda()], dim=0) # (2)'''

    def forward(self, x):
        '''route = self.router(x)
        x = torch.einsum('kdj, bjf -> kbdf', torch.softmax(self.E2, dim=-1), x)
        x = torch.einsum('kid, kbdf -> kbif', torch.softmax(self.E1, dim=-1), x)
        x = torch.einsum('bik, kbif -> bif', route, x)
        x = torch.einsum('bik, e, ekid, ekdj, bjf -> bif', 
                         self.router(x),
                         self.cal_coefficient(),
                         torch.softmax(self.E1, dim=-1),
                         torch.softmax(self.E2, dim=-1),
                         x)'''
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = torch.cat(torch.split(q, self.head_dim, dim=-1), dim=0)
        k = torch.cat(torch.split(k, self.head_dim, dim=-1), dim=0)
        v = torch.cat(torch.split(v, self.head_dim, dim=-1), dim=0)
        attn = torch.softmax(q @ k.transpose(-1, -2) / self.head_dim*0.5, 
                             dim=-1)
        if False:
            v = (attn - attn @ attn) @ v
        else:
            v = attn @ v
        v = torch.cat(torch.split(v, x.shape[0], dim=0), dim=-1)
        v = self.mh(v)
        return v
    
class MAGE_Block(nn.Module):
    def __init__(self, dim, expert_num, node_num, depth, model_dim, graph_gen_dim=32, head=4):
        super(MAGE_Block, self).__init__()
        # Differential
        self.E1 = nn.Parameter(torch.rand(2, expert_num, node_num, graph_gen_dim))
        self.E2 = nn.Parameter(torch.rand(2, expert_num, graph_gen_dim, node_num))
        self.router = nn.Sequential(
            nn.Linear(dim, expert_num),
            nn.Softmax(dim=-1)
        )
        
        self.lambda_init = self.lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.coefficient = nn.Parameter(torch.ones(1), requires_grad=False)

    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

    def cal_lambda(self):
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1, keepdim=True).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1, keepdim=True).float())
        lambda_full = - (lambda_1 - lambda_2 + self.lambda_init)
        return lambda_full 

    def cal_coefficient(self):
        return torch.concat([self.coefficient, self.cal_lambda()], dim=0) # (2)

    def forward(self, x):
        '''route = self.router(x)
        x = torch.einsum('kdj, bjf -> kbdf', torch.softmax(self.E2, dim=-1), x)
        x = torch.einsum('kid, kbdf -> kbif', torch.softmax(self.E1, dim=-1), x)
        x = torch.einsum('bik, kbif -> bif', route, x)'''
        x = torch.einsum('bik, e, ekid, ekdj, bjf -> bif', 
                         self.router(x),
                         self.cal_coefficient(),
                         torch.softmax(self.E1, dim=-1),
                         torch.softmax(self.E2, dim=-1),
                         x)
        return x


class SwiGLU_FFN(nn.Module):
    def __init__(self, dim_in, dim_out, expand_ratio=4, dropout=0.3, norm=None):
        super(SwiGLU_FFN, self).__init__()
        self.W1 = nn.Linear(dim_in, expand_ratio*dim_in)
        self.W2 = nn.Linear(dim_in, expand_ratio*dim_in)
        self.W3 = nn.Linear(expand_ratio*dim_in, dim_out)
        
        self.dropout =  nn.Dropout(dropout)
        
    def forward(self, x):
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
    

class Prompt_Pool(nn.Module):
    def __init__(self, model_args):
        super(Prompt_Pool, self).__init__()
        self.feature_dim = model_args.feature_dim

        self.spatial_prompt = nn.init.xavier_uniform_(nn.Parameter(
            torch.empty(model_args.node_num, model_args.model_dim)))

        print('Temporal Prompt Pool has:')
        self.pool = nn.ModuleList()
        self.denorm = []
        if model_args.second:
            self.pool.append(nn.Embedding(model_args.second, model_args.model_dim))
            self.denorm.append(model_args.second)
            print('second')
        if model_args.minute:
            self.pool.append(nn.Embedding(model_args.minute, model_args.model_dim))
            self.denorm.append(model_args.minute)
            print('minute')
        if model_args.hour:
            self.pool.append(nn.Embedding(model_args.hour, model_args.model_dim))
            self.denorm.append(model_args.hour)
            print('hour')
        if model_args.day:
            self.pool.append(nn.Embedding(model_args.day, model_args.model_dim))
            self.denorm.append(model_args.day)
            print('day')
        if model_args.week:
            self.pool.append(nn.Embedding(model_args.week, model_args.model_dim))
            self.denorm.append(model_args.week)
            print('week')
        if model_args.weekday:
            self.pool.append(nn.Embedding(model_args.weekday, model_args.model_dim))
            self.denorm.append(model_args.weekday)
            print('weekday')
        if model_args.month:
            self.pool.append(nn.Embedding(model_args.month, model_args.model_dim))
            self.denorm.append(model_args.month)
            print('month')
        if model_args.quarter:
            self.pool.append(nn.Embedding(model_args.quarter, model_args.model_dim))
            self.denorm.append(model_args.quarter)
            print('quarter')
        if model_args.year:
            self.pool.append(nn.Embedding(model_args.year, model_args.model_dim))
            self.denorm.append(model_args.year)
            print('year')

    def forward(self, temporal):
        prompt = 0
        for i in range(len(self.pool)):
            prompt += self.pool[i]((temporal[:, -1, :, i+self.feature_dim] * self.denorm[i]).long())
        return prompt + self.spatial_prompt