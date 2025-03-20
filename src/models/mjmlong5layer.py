import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from src.base.model import BaseModel
from src.utils.alpha_entmax import Entmax15
import sys
CUDA_LAUNCH_BLOCKING=1

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class STID(BaseModel):
    def __init__(self, layer, **args):
        super(STID, self).__init__(**args)
        # attributes
        self.num_nodes = self.node_num
        self.node_dim = 32
        self.input_len = 12
        self.input_dim = 4
        self.embed_dim = 32 # 64
        self.output_len = 12
        self.num_layer = layer
        self.temp_dim_tid = 32
        self.temp_dim_diw = 32
        self.temp_dim_miy = 32
        '''
        self.num_nodes = self.node_num
        self.node_dim = 32
        self.input_len = 12
        self.input_dim = 2
        self.embed_dim = 32 # 64
        self.output_len = 12
        self.num_layer = 3
        self.temp_dim_tid = 32
        self.temp_dim_diw = 32
        '''
        self.time_of_day_size = 96 # 288
        self.day_of_week_size = 7
        self.month_of_year_size = 12
        self.if_time_in_day = 1
        self.if_day_in_week = 1
        self.if_spatial = 1
        self.if_month_in_year = 0

        # spatial embeddings
        # if self.if_spatial:
        #     self.node_emb = nn.Parameter(
        #         torch.empty(self.input_len, self.num_nodes, self.node_dim))
        #     nn.init.xavier_uniform_(self.node_emb)
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            print('day')
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            print('week')
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        if self.if_month_in_year:
            print('month')
            self.month_in_year_emb = nn.Parameter(
                torch.empty(self.month_of_year_size, self.temp_dim_miy))
            nn.init.xavier_uniform_(self.month_in_year_emb)
            
        '''
        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        '''
        # decomposing
        kernel_size = 3
        self.decompsition = series_decomp(kernel_size)
        
        self.time_series_emb_layer1 = nn.Conv1d(
            in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=1, bias=True)
        self.time_series_emb_layer2 = nn.Conv1d(
            in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=1, bias=True)
        # nn.init.xavier_uniform_(self.time_series_emb_layer1)
        # nn.init.xavier_uniform_(self.time_series_emb_layer2)
        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial) + self.temp_dim_tid*int(self.if_day_in_week) + \
            self.temp_dim_diw*int(self.if_time_in_day) + \
            self.temp_dim_miy*int(self.if_month_in_year)
        self.encoder = nn.Sequential(
            *[STIDMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
    
    def forward(self, history_data, label=None, adj=None):
        # prepare data
        # input_data = history_data
        # print(input_data.shape)
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            # print(t_i_d_data)
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            # print(d_i_w_data)
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]
        else:
            day_in_week_emb = None
        if self.if_month_in_year:
            m_i_y_data = history_data[..., 3]
            # print(d_i_w_data)
            # a = (m_i_y_data[:, -1, :] * self.month_of_year_size).type(torch.LongTensor)
            # print(max(a), min(a))
            # sys.exit()
            month_in_year_emb = self.month_in_year_emb[(m_i_y_data[:, -1, :] * self.month_of_year_size - 1).type(torch.LongTensor)]
        else:
            month_in_year_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = history_data.shape
        '''
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1) # (b, n, t*f) -> (b, t*f, n, 1)
        time_series_emb = self.time_series_emb_layer(input_data)
        '''
        
        seasonal_init, trend_init = self.decompsition(history_data[..., 0])
        # print(seasonal_init.shape)
        # seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        seasonal_output = self.time_series_emb_layer1(seasonal_init)
        trend_output = self.time_series_emb_layer2(trend_init)

        time_series_emb = (seasonal_output + trend_output).unsqueeze(-1)

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))
        if month_in_year_emb is not None:
            tem_emb.append(month_in_year_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        h = hidden.transpose(1, -1)
        # encoding
        if self.num_layer:
            hidden = self.encoder(hidden)
        z = hidden.transpose(1, -1)
        # regression
        prediction = self.regression_layer(hidden)

        return  h, z, prediction
    
    def module(self, hidden, label=None, adj=None):
        if self.num_layer:
        # encoding
            hidden = self.encoder(hidden.transpose(1, -1)).transpose(1, -1) # (b, d, n, 1)

        return hidden 
    

class STIDMultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=4*hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=4*hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        # self.act = nn.ReLU()
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden


class STRIP(BaseModel):
    def __init__(self, model_args, stmodel, dim, core, **args):
        super(STRIP, self).__init__(**args)
        ## base spatio-temporal model
        self.stmodel = stmodel
        # self.inverse = stmodel.if_inverse()

        ## training type parameters
        self.extra_type = model_args['extra_type'] # 0: baseline, 1: baseline + strip joint-training, 2: baseline (fixed) + strip fine-tuning
        self.same = model_args['same']
        if self.extra_type and not self.same:
            self.stmodel_detach = copy.deepcopy(stmodel)

        ## datasets parameters
        # self.node_num = model_args['node_num']

        ## backcast parameters and networks
        self.in_dim = dim[0] # equals to the dim of hidden emb before entering decoder/predictor/out-put layer.
        self.out_dim = dim[1] # equals to the dim of shallow emb. 
        self.backcast_hidden_dim = model_args['hid_dim'] # the dim in the backcast network. 
        if core:
            self.backcast = Core_Adaptive(self.in_dim, self.in_dim, self.out_dim, self.node_num, core)
        else:
            self.backcast = nn.Sequential(
                nn.Linear(self.in_dim, 4*self.in_dim), 
                nn.GELU(),
                nn.Linear(4*self.in_dim, self.out_dim), 
            )

        ## decoder parameters and nerworks
        self.decoder_hidden_dim = model_args['hid_dim'] # the dim in the decoder network. 
        self.horizon = model_args['horizon'] # the horizon
        self.decoder = nn.Sequential(
            nn.Linear(self.in_dim, self.decoder_hidden_dim), 
            nn.GELU(),
            # nn.Dropout(0.15),
            # nn.Linear(self.decoder_hidden_dim, self.horizon if self.inverse else self.output_dim),
            nn.Linear(self.decoder_hidden_dim, self.horizon),  
        )

        ## residual propagation parameters and networks 
        ### parameters for kernel.
        self.adj = model_args['predefined_adj']
        self.kernel_list = model_args['adjs'] if self.adj else [] # distance or longtitude-latitude adjacency matrices
        self.adj_num = len(self.kernel_list) # equals to len(kernel_list), or adj numbers or supports number or sth else. 
        self.datadriven_adj_num = 0
        self.adptive_adj_num = 0
        self.rp_layer = model_args['rp_layer'] # layers of residual propagation if discrete
        ### parameters for adaptive and data-driven residual correlation graphs. 
        self.datadriven_adj = model_args['datadriven_adj']
        self.datadriven_adj_dim = model_args['datadriven_adj_dim']
        self.datadriven_adj_head = model_args['datadriven_adj_head']
        self.adaptive_adj_dim = model_args['adaptive_adj_dim']
        self.adaptive_adj = model_args['adaptive_adj']
        self.mrf = model_args['mrf']
        if self.datadriven_adj: 
            self.datadriven_adj_num += 1
            self.Q = nn.Linear(self.in_dim, 
                               self.datadriven_adj_dim)
            if not self.mrf: 
                self.K = nn.Linear(self.in_dim, 
                                   self.datadriven_adj_dim)
        if self.adaptive_adj: 
            self.adptive_adj_num += 1
            self.E1 = nn.Parameter(torch.randn(self.node_num, 
                                               self.adaptive_adj_dim), 
                                   requires_grad=True)
            if not self.mrf: 
                self.E2 = nn.Parameter(torch.randn(self.node_num, 
                                                   self.adaptive_adj_dim), 
                                       requires_grad=True)
        self.kernel_num = self.adj_num + self.datadriven_adj_num + self.adptive_adj_num

        ### parameters for alpha and beta (beta is the gamma in paper)
        self.use_global_opt = model_args['use_global_opt']
        if self.kernel_num: 
            if self.use_global_opt:
                self.alpha = nn.Parameter(torch.FloatTensor(1)).unsqueeze(-1).repeat(1, self.node_num) # (k, n)
                self.beta = nn.Parameter(torch.FloatTensor(1)).unsqueeze(-1).repeat(1, self.node_num) # (n)
            else:
                self.alpha = nn.Parameter(torch.FloatTensor(self.kernel_num, self.node_num)) # (k, n)
                self.beta = nn.Parameter(torch.FloatTensor(1, self.node_num)) # (n)
            self.reset_parameters()
            self.alpha_activation = nn.Tanh()
            self.beta_activation = nn.Sigmoid()

    def reset_parameters(self):
         nn.init.uniform_(self.alpha)
         nn.init.uniform_(self.beta)
    
    def forward(self, x, label=None): 
        if self.extra_type:
            h, z, y = self.stmodel(x, label)

            h_res = self.backcast(z)
            z_res = self.stmodel.module(h - h_res, label) if self.same else self.stmodel_detach.module(h - h_res, label)
            z_res = self.residual_information_propagation(z_res)
            y_res = self.decoder(z_res)
            return y + y_res.transpose(1, -1)
        else:
            return self.stmodel(x, label)[-1]
        
        
    def residual_information_propagation(self, z):
        if self.kernel_num: 
            alpha = self.alpha_activation(self.alpha)
            beta = self.beta_activation(self.beta)
            kernel_list = self.kernel_list

            i = 0
            kernel = torch.tensor(0).to(z.device)
            if self.adj_num:
                # kernel_pre = alpha[i] * kernel_list[0] + alpha[i+1] * kernel_list[1]
                kernel = kernel + alpha[i] * kernel_list[0] + alpha[i+1] * kernel_list[1]
                i = i+2
            if self.adaptive_adj:
                # kernel_ad = alpha[i] * self.adaptive_adj_generation()
                kernel = kernel + alpha[i] * self.adaptive_adj_generation()
                i += 1
            if self.datadriven_adj:
                # kernel_da = alpha[i] * self.datadriven_adj_generation(z)
                kernel = kernel.unsqueeze(0).unsqueeze(0) + alpha[i] * self.datadriven_adj_generation(z)
            
            kernel = beta * (self.kernel_num * torch.eye(self.node_num).to(kernel.device) + kernel) # (k, n, n)
            z = self.nconv(z, kernel)
        return z
    
    def nconv(self, x, kernel): # x (b, t, n, f), kernel (k, n, n)
        for _ in range(self.rp_layer):
            if self.datadriven_adj:
                # print(kernel.shape)
                x = torch.einsum('btnf, btmn -> btmf', x, kernel)
            else:
                x = torch.einsum('btnf, mn -> btmf', x, kernel)
        return x.contiguous()
    
    def datadriven_adj_generation(self, x):
        Q = self.Q(x)
        if not self.mrf:
            K = self.K(x)

        datadriven_kernel = torch.einsum('btif, btjf -> btij', Q, Q if self.mrf else K)
        diag = torch.arange(self.node_num)
        datadriven_kernel[..., diag, diag] = 0
        datadriven_kernel = datadriven_kernel / math.sqrt(self.datadriven_adj_dim)
        datadriven_kernel = torch.softmax(datadriven_kernel, dim=-1)
        return datadriven_kernel
    
    def adaptive_adj_generation(self):
        adaptive_kernel = torch.einsum('id, jd -> ij', self.E1, self.E1 if self.mrf else self.E2)
        adaptive_kernel = torch.softmax(torch.relu(adaptive_kernel.fill_diagonal_(0)), dim=-1)
        return adaptive_kernel
    

class Core_Adaptive(nn.Module):
    def __init__(self, d_in, d_core, d_out, node_num, core_num, nndropout=0.3, dropout=0.08):
        super(Core_Adaptive, self).__init__()
        if core_num == 0:
            raise 
        else:
            print('core number is', core_num)
        self.node_num = node_num
        self.adpative = nn.Parameter(torch.randn(d_core, node_num))
        self.cores = nn.Parameter(torch.randn(core_num, d_core))
        self.affiliation = nn.Parameter(torch.randn(core_num, node_num))
        # nn.init.xavier_uniform_(self.adpative)
        # nn.init.xavier_uniform_(self.cores)
        # nn.init.xavier_uniform_(self.affiliation)

        self.value = nn.Conv2d(d_in, d_core, kernel_size=(1, 1))
        self.ffn = nn.Sequential(
            nn.Conv2d(d_in + d_core, 4*(d_in + d_core), kernel_size=(1, 1)),
            nn.GELU(),
            nn.Dropout(nndropout),
            nn.Conv2d(4*(d_in + d_core), d_out, kernel_size=(1, 1)),
        )
        
        self.d_core = d_core
        self.core_num = core_num

        self.nndropout = nn.Dropout(nndropout)
        self.dropout = dropout
        self.norm = nn.BatchNorm2d(d_out)
        # self.norm = nn.LayerNorm(d_out)
        # self.entmax_n2c = Entmax15(dim=1)
        # self.entmax_c2n = Entmax15(dim=0)


    def forward(self, input, adj=None, *args, **kwargs): 
        # input: (b, f, t, n)
        input = input.permute(0, 3, 1, 2)
        affiliation = self.cores @ self.adpative / self.d_core**0.5 # (c, n)
        
        # affiliation = self.affiliation
        affiliation_node_to_core = torch.softmax(affiliation, dim=1) # (c, n)
        affiliation_core_to_node = torch.softmax(affiliation, dim=0) # (c, n)
        # affiliation_node_to_core = self.entmax_n2c(affiliation) # (c, n)
        # affiliation_core_to_node = self.entmax_c2n(affiliation) # (c, n)
        
        '''
        np.save('core_emb.npy', self.cores.detach().cpu().numpy())
        np.save('node_emb.npy', self.adpative.detach().cpu().numpy())
        np.save('affiliation.npy', affiliation.detach().cpu().numpy())
        np.save('affiliation_node_to_core.npy', affiliation_node_to_core.detach().cpu().numpy())
        np.save('affiliation_core_to_node.npy', affiliation_core_to_node.detach().cpu().numpy())
        print("Done!")
        sys.exit()
        '''
        
        
        v = self.value(input)
        v = torch.einsum('bftn, cn -> bftc', v, affiliation_node_to_core)
        v = torch.einsum('bftc, cn -> bftn', v, affiliation_core_to_node)
        output = torch.cat([input-v, v], dim=1)
        output = self.ffn(output)
        output = output + input
        # output = self.norm(output.transpose(1, -1)).transpose(1, -1)
        output = self.norm(output)
        return output.permute(0, 2, 3, 1)



