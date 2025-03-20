import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from torch.nn.parameter import Parameter
from einops import rearrange
# from torch_geometric.nn import HypergraphConv
from torch_geometric.nn import DenseGATConv
from torch.nn import init
import numbers
from src.base.model import BaseModel

class t_multi_scale_construct3(nn.Module):
    def __init__(self, dim_in=32, dim_out=32, temporal_pooling_ratio=2, n_temporal_scales=3):
        super(t_multi_scale_construct3, self).__init__()
        self.r = temporal_pooling_ratio
        self.n_temporal_scales = n_temporal_scales
        self.temporal_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        for i in range(n_temporal_scales-1):
            self.temporal_layers.append(nn.Conv2d(dim_in,dim_out,(1,1),stride=(1,1), dilation=(1,1)))
            self.pooling_layers.append(nn.AvgPool2d(kernel_size=(1,self.r), stride=(1,self.r)))

    def forward(self, x):
        x_list = [x]
        x_temp = x
        for i in range(0, self.n_temporal_scales-1):
            x_temp = self.temporal_layers[i](x_temp)
            x_temp = self.pooling_layers[i](x_temp)
            x_list.append(x_temp)
        return x_list


class temporal_proj(nn.Module):
    def __init__(self, dim_in=32, dim_out=32,proj_ratio=4, num_temporal_scales=3):
        super(temporal_proj, self).__init__()
        self.t_conv_layer_1 = nn.Conv2d(dim_in,dim_out,(1,1),stride=(1,1), dilation=(1,1))
        self.t_pool_1 = nn.AvgPool2d(kernel_size=(1,4), stride=(1,4))
        self.t_conv_layer_2 = nn.Conv2d(dim_in,dim_out,(1,1),stride=(1,1), dilation=(1,1))
        self.t_pool_2 = nn.AvgPool2d(kernel_size=(1,4), stride=(1,4))
        self.t_conv_layer_3 = nn.Conv2d(dim_in,dim_out,(1,1),stride=(1,1), dilation=(1,1))
        self.t_pool_3 = nn.AvgPool2d(kernel_size=(1,4), stride=(1,4))

    def forward(self, x_list):
        x_list[0] = self.t_conv_layer_1(x_list[0])
        x_list[0] = self.t_pool_1(x_list[0])
        x_list[1] = self.t_conv_layer_2(x_list[1])
        x_list[1] = self.t_pool_2(x_list[1])
        x_list[2] = self.t_conv_layer_3(x_list[2])
        x_list[2] = self.t_pool_3(x_list[2])
        return x_list


class MSJ_STHyper(BaseModel):
    def __init__(self, num_nodes, num_spatial_scale, graph_pooling_ratio, num_temporal_scale, temporal_pooling_ratio, num_hyperedges, num_hyper_layers,
                 input_len, output_len, iinput_dim, ooutput_dim, rnn_units, num_layers=1, cheb_k=3, task='short_term',
                 ycov_dim=1, mem_num=20, mem_dim=32, cl_decay_steps=2000, use_curriculum_learning=False, device='cuda:0', **args):
        super(MSJ_STHyper, self).__init__(**args)
        self.num_nodes = num_nodes
        self.num_spatial_scales = num_spatial_scale
        self.p = graph_pooling_ratio
        self.num_temporal_scales = num_temporal_scale
        self.r = temporal_pooling_ratio
        self.num_hyperedges = num_hyperedges
        self.n_hyper_layers = num_hyper_layers
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = iinput_dim
        self.rnn_units = rnn_units
        self.output_dim = ooutput_dim
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.device = device
        self.all_spatial_nodes = self.num_nodes
        self.spatial_nodes_list = [self.num_nodes]
        self.task = task

        # Declare assignment mateixs
        self.assign_mx_list = nn.ParameterList()
        for i in range(self.num_spatial_scales-1):
            n_cur_spatial_scale = int(self.num_nodes/(self.p ** (i)))
            n_next_spatial_scale = int(n_cur_spatial_scale/self.p)
            self.all_spatial_nodes += n_next_spatial_scale
            self.spatial_nodes_list.append(n_next_spatial_scale)
            temp_s = Parameter(torch.randn(n_cur_spatial_scale, n_next_spatial_scale).to(device, non_blocking=True), requires_grad=True)
            self.assign_mx_list.append(temp_s)

        # # Extract temporal features
        self.t_multi_scale_construct_layer1 = t_multi_scale_construct3(dim_in=self.input_dim, dim_out=self.input_dim,
                                                                       n_temporal_scales=self.num_temporal_scales, temporal_pooling_ratio=self.r)
        # self.t_multi_scale_construct_layer2 = t_multi_scale_construct4(dim_in=self.rnn_units, dim_out=self.rnn_units)

        # memory
        self.hidden_dim = self.rnn_units * 2
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory_list = nn.ModuleList()
        for i in range(self.num_spatial_scales):
            self.memory_list.append(self.construct_memory(self.spatial_nodes_list[i], self.rnn_units))
        self.memory_h = self.construct_memory(self.num_hyperedges, self.hidden_dim)

        # encoder
        self.encoder_list = nn.ModuleList()
        self.encoder_list.append(
        ADCRNN_Encoder(self.spatial_nodes_list[0], self.input_dim, self.rnn_units, self.cheb_k, self.num_layers,
                       num_of_time_scales=self.num_temporal_scales))
        for i in range(1, self.num_spatial_scales):
            self.encoder_list.append(
                ADCRNN_Encoder(self.spatial_nodes_list[i], self.rnn_units, self.rnn_units, self.cheb_k, self.num_layers,
                               num_of_time_scales=self.num_temporal_scales))


        # decoder
        self.decoder_dim = (self.rnn_units + self.mem_dim)
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim, self.decoder_dim, self.cheb_k, self.num_layers)

        # Hypergraph
        self.hypergnn = nn.ModuleList()
        self.n_st_nodes = self.all_spatial_nodes * self.num_temporal_scales
        self.hyper_incidence_mx = nn.Parameter(torch.randn(self.n_st_nodes, self.num_hyperedges), requires_grad=True)
        self.U = nn.Parameter(torch.randn(self.num_hyperedges, self.num_hyperedges), requires_grad=True)
        self.hyper_mlp1 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim, bias=True), nn.ReLU(inplace=True))
        self.st_layernrom = LayerNorm(normalized_shape=(self.hidden_dim, self.n_st_nodes, 1), elementwise_affine=False)
        self.dense_gat_hyedge_1 = DenseGATConv(in_channels=self.hidden_dim, out_channels=self.mem_dim, dropout=0.3, heads=2)
        self.proj_mlp1 = nn.Sequential(nn.Linear(self.hidden_dim + self.mem_dim, self.hidden_dim, bias=True), nn.ReLU(inplace=True))

        # fusion
        self.wj_list = nn.ParameterList()
        for i in range(self.num_spatial_scales):
            self.wj_list.append(Parameter(torch.randn(self.num_temporal_scales).to(device, non_blocking=True), requires_grad=True))

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
        if self.task == 'long_term':
            self.outputmlp = nn.Sequential(nn.Linear(self.decoder_dim, self.decoder_dim, bias=True),
                                           nn.ReLU(inplace=True))

    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self, num_nodes, hidden_dim):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(hidden_dim, self.mem_dim), requires_grad=True)    # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(num_nodes, self.mem_num), requires_grad=True) # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(num_nodes, self.mem_num), requires_grad=True) # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t:torch.Tensor, memory_index):
        if memory_index == 'hyper':
            memory = self.memory_h
        else:
            memory = self.memory_list[memory_index]
        query = torch.matmul(h_t, memory['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, memory['Memory'].t()), dim=-1)         # alpha: (B, N, M)
        value = torch.matmul(att_score, memory['Memory'])     # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = memory['Memory'][ind[:, :, 0]] # B, N, d
        neg = memory['Memory'][ind[:, :, 1]] # B, N, d
        return value, query, pos, neg


    def hypergnn(self, h_st, supports=None, supports_cross=None):
        # h_st: (B, N, d)
        hyper_incidence_mx = F.softmax(self.hyper_incidence_mx, dim=-1)
        mask = torch.zeros(hyper_incidence_mx.size(0), hyper_incidence_mx.size(1)).to(h_st.device)
        mask.fill_(float('0'))
        s1, t1 = (hyper_incidence_mx + torch.rand_like(hyper_incidence_mx) * 0.01).topk(20, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        hyper_incidence_mx = hyper_incidence_mx * mask

        hyper_incidence_mx_T = rearrange(hyper_incidence_mx, 'n i -> i n')

        U = F.softmax(self.U)
        e1 = torch.einsum('in,bnc->bic', [hyper_incidence_mx_T, h_st])
        e2 = F.leaky_relu(torch.einsum('ii,bic->bic', [U, e1]))

        # hyperedge propagation
        node_embeddings1_h = torch.matmul(self.memory_h['We1'], self.memory_h['Memory'])
        node_embeddings2_h = torch.matmul(self.memory_h['We2'], self.memory_h['Memory'])
        U3 = F.softmax(F.relu(torch.mm(node_embeddings1_h, node_embeddings2_h.T)), dim=-1)

        e3 = self.dense_gat_hyedge_1(e1, U3, add_loop=True)
        e3_att, _, _, _ = self.query_memory(e3, 'hyper')
        e3 = torch.cat([e3, e3_att], dim=-1)
        e3 = self.proj_mlp1(e3)

        hyperedge_embedding = e1 + e2 + e3

        # MLP
        hyperedge_embedding = self.hyper_mlp1(hyperedge_embedding)

        node_embedding = torch.einsum('ni,bic->bnc', [hyper_incidence_mx, hyperedge_embedding])
        node_embedding = self.st_layernrom(node_embedding)
        node_embedding = torch.tanh(h_st) * torch.sigmoid(node_embedding)

        return node_embedding


    def forward(self, x, labels=None, batches_seen=None):

        # x: B, T, N, hidden
        # Extract multi-scale temproal features
        x = rearrange(x, 'b t n c -> b c n t')
        x_list = self.t_multi_scale_construct_layer1(x)
        for i in range(self.num_temporal_scales):
            x_list[i] = rearrange(x_list[i], 'b c n t -> b t n c')

        # STPM moudle
        output_STPM = []
        supports = []
        h_att_list = []
        query_list = []
        pos_list = []
        neg_list = []
        for i in range(self.num_spatial_scales):
            if i != self.num_spatial_scales-1:
                ass = F.softmax(self.assign_mx_list[i],dim=1)
                ass_T = rearrange(ass, 'n m -> m n')

            node_embeddings1 = torch.matmul(self.memory_list[i]['We1'], self.memory_list[i]['Memory'])
            node_embeddings2 = torch.matmul(self.memory_list[i]['We2'], self.memory_list[i]['Memory'])
            g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
            g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
            cur_supports = [g1, g2]
            supports.append(cur_supports)

            # st encoder for spatial i
            init_state = self.encoder_list[i].init_hidden(x.shape[0])
            h_en_list, last_states = self.encoder_list[i](x_list, init_state, cur_supports)  # B, T, N, hidden
            # assign hidden states of spatial scale i to next spatial scale
            if i != self.num_spatial_scales-1:
                x_list = []
                for j in range(self.num_temporal_scales):
                    x_list.append(torch.einsum("btnc, mn->btmc", [h_en_list[j], ass_T]))
            # get memory-augmented features
            for j in range(self.num_temporal_scales):
                h_t = last_states[j]
                h_att, query, pos, neg = self.query_memory(h_t, memory_index=i)
                h_att_list.append(h_att)
                query_list.append(query)
                pos_list.append(pos)
                neg_list.append(neg)
                h_t_plus = torch.cat([h_t, h_att], dim=-1)
                output_STPM.append(h_t_plus)


        # AHM module
        output_STPM = torch.cat(output_STPM, dim=1)
        node_embedding = self.hypergnn(output_STPM)

        # Fusion module
        # Temporal fusion
        x_t = []
        cur_node_index = 0
        for i in range(self.num_spatial_scales):
            cur_nodes = self.spatial_nodes_list[i]
            wj = F.softmax(self.wj_list[i], dim=-1)
            # temporal scale from 0 to k
            t_fused = node_embedding[:,cur_node_index:cur_node_index+cur_nodes,:]* wj[0]
            for j in range(1, self.num_temporal_scales):
                t_fused += node_embedding[:,cur_node_index:cur_node_index+cur_nodes,:]* wj[j]
                cur_node_index += cur_nodes
            x_t.append(t_fused)

        # Spatial fusion
        final_rep = x_t[0]
        for i in range(1, self.num_spatial_scales):
            assign_matrices_prod = F.softmax(self.assign_mx_list[0], dim=-1)
            if i == 1:
                final_rep += torch.einsum("bqc, nq->bnc", [x_t[1], assign_matrices_prod])
            else:
                for j in range(1, i):
                    cur_ass = F.softmax(self.assign_mx_list[j], dim=-1)
                    assign_matrices_prod = torch.einsum("nm, mq->nq", [assign_matrices_prod, cur_ass])
                final_rep += torch.einsum("bqc, nq->bnc", [x_t[2], assign_matrices_prod])


        # Output module
        if self.task == 'short_term':
            ht_list = [final_rep]
            go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
            out = []
            for t in range(self.output_len):
                h_de, ht_list = self.decoder(go, ht_list, supports[0])
                go = self.proj(h_de)
                out.append(go)
                if self.training and self.use_curriculum_learning:
                    c = np.random.uniform(0, 1)
                    if c < self.compute_sampling_threshold(batches_seen):
                        go = labels[:, t, ...]
            output = torch.stack(out, dim=1)
        elif self.task == 'long_term':
            x = self.outputmlp(final_rep)
            output = self.proj(x)
            output = output.unsqueeze(-1)
            output = rearrange(output, 'b n t c -> b t n c')

        return output, h_att_list[0], query_list[0], pos_list[0], neg_list[0], supports, self.assign_mx_list


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx=None):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
    
class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, support_len):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(
            torch.FloatTensor(support_len * cheb_k * dim_in, dim_out))  # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)  # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, support_len=2)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k, support_len=2)

    def forward(self, x, state, supports):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers, is_t_multiple_scale=True, num_of_time_scales=3):
        super(ADCRNN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_of_time_scales):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        self.is_t_multiple_scale = is_t_multiple_scale
        self.num_of_time_scales = num_of_time_scales

    def forward(self, x_list, init_state, supports):
        # shape of x: (B, T, N, D), shape of init_state: (num_layers, B, N, hidden_dim)
        # assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        # seq_length = x.shape[1]
        # current_inputs = x
        output_hidden = []
        current_inputs = []
        if self.is_t_multiple_scale:
            for i in range(self.num_of_time_scales):
                current_inputs.append(x_list[i])
                state = init_state[i]
                inner_states = []
                seq_length = current_inputs[i].shape[1]
                for t in range(seq_length):
                    state = self.dcrnn_cells[i](current_inputs[i][:, t, :, :], state, supports)
                    inner_states.append(state)
                output_hidden.append(state)
                current_inputs[i] = torch.stack(inner_states, dim=1)

        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        if self.is_t_multiple_scale:
            for i in range(self.num_of_time_scales):
                init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        else:
            for i in range(self.num_layers):
                init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states


class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers, is_t_multiple_scale=True, num_of_time_scales=1):
        super(ADCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_of_time_scales):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))
        self.is_t_multiple_scale = is_t_multiple_scale
        self.num_of_time_scales = num_of_time_scales

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        # print(xt.shape,self.node_num,self.input_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim

        current_inputs = xt
        output_hidden = []
        for i in range(self.num_of_time_scales):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden
