import torch
import torch.nn as nn
import torch.nn.functional as F

def if_symmetric(adj):
    n, m = adj.shape
    if n != m:
        raise ValueError("Input adj must be square.")

    max_diff = torch.max(torch.abs(adj - adj.t()))

    eps = torch.finfo(adj.dtype).eps

    return (max_diff < eps)

def determine_normalization(matrix):
    first_row = matrix[0, :]
    first_col = matrix[:, 0]

    row_normalized = (first_row.sum() == 1)
    col_normalized = (first_col.sum() == 1)

    if row_normalized:
        print("row_normalized")
        return matrix
    elif col_normalized:
        print("col_normalized, need transpose!")
        return matrix.T
    else:
        raise print("Error: no normalization")

class Core_Predefined(nn.Module):
    def __init__(self, d_in, d_core, d_out, node_num, core_num, adj=None, nndropout=0.3, dropout=0.08):
        super(Core_Predefined, self).__init__()
        if core_num == 0:
            raise 
        else:
            print('core number is', core_num)
        self.node_num = node_num

        self.align = nn.Linear(node_num, d_core)
        self.adj = adj if adj is not None else None
        self.cores = nn.Parameter(torch.randn(core_num, d_core))
        nn.init.xavier_uniform_(self.cores)

        self.value = nn.Conv2d(d_in, d_core, kernel_size=(1, 1))
        self.ffn = nn.Sequential(
            nn.Conv2d(d_in + d_core, 4*(d_in + d_core), kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(4*(d_in + d_core), d_out, kernel_size=(1, 1)),
        )
        
        self.d_core = d_core
        self.core_num = core_num

        self.nndropout = nn.Dropout(nndropout)
        self.dropout = dropout
        self.ln = nn.LayerNorm(d_out)

    def forward(self, input, adj=None, *args, **kwargs): 
        # input: (b, f, t, n)
        if adj is None:
            adj = self.adj
        affiliation = self.cores @ self.align(adj).T / self.d_core**0.5 # (c, n)
        affiliation_node_to_core = torch.softmax(affiliation, dim=1) # (c, n)
        affiliation_core_to_node = torch.softmax(affiliation, dim=0) # (c, n)
        # print(affiliation.shape, affiliation_node_to_core.shape, affiliation_core_to_node.shape)

        v = self.value(input)
        if self.training:
            mask = torch.zeros_like(affiliation, requires_grad=False) # (c, n)
            indices = torch.multinomial(affiliation_node_to_core, int((1-self.dropout)*self.node_num)) # (c, n) -> (c, m)
            # print(indices.shape)
            # print(mask.shape)
            # mask[indices] = 1
            mask.scatter_(1, indices, 1)
            affiliation_node_to_core = torch.softmax(affiliation + mask.log(), dim=1) # (c, n)
            v = torch.einsum('bftn, cn -> bftc', v, affiliation_node_to_core)
            v = torch.einsum('bftc, cn -> bftn', v, affiliation_core_to_node)
        else:
            v = torch.einsum('bftn, cn -> bftc', v, affiliation_node_to_core)
            v = torch.einsum('bftc, cn -> bftn', v, affiliation_core_to_node)
        
        output = torch.cat([input-v, v], 1)
        output = self.ffn(output) # (b, f, t, n)
        # print(output.shape, input.shape)
        # output = input + self.nndropout(output)
        # output = self.ln(output.transpose(1, -1)).transpose(1, -1)

        return output
    


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
        nn.init.xavier_uniform_(self.adpative)
        nn.init.xavier_uniform_(self.cores)

        self.value = nn.Conv2d(d_in, d_core, kernel_size=(1, 1))
        self.ffn = nn.Sequential(
            nn.Conv2d(d_in + d_core, 4*(d_in + d_core), kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(4*(d_in + d_core), d_out, kernel_size=(1, 1)),
        )
        
        self.d_core = d_core
        self.core_num = core_num

        self.nndropout = nn.Dropout(nndropout)
        self.dropout = dropout
        self.ln = nn.LayerNorm(d_out)

    def forward(self, input, adj=None, *args, **kwargs): 
        # input: (b, f, t, n)
        affiliation = self.cores @ self.adpative / self.d_core**0.5 # (c, n)
        affiliation_node_to_core = torch.softmax(affiliation, dim=1) # (c, n)
        affiliation_core_to_node = torch.softmax(affiliation, dim=0) # (c, n)

        v = self.value(input)
        v = torch.einsum('bftn, cn -> bftc', v, affiliation_node_to_core)
        v = torch.einsum('bftc, cn -> bftn', v, affiliation_core_to_node)
        output = torch.cat([input-v, v], dim=1)
        output = self.ffn(output)
        return output


class Core_Datadriven(nn.Module):
    def __init__(self, d_in, d_core, d_out, node_num, core_num, nndropout=0.3, dropout=0.08):
        super(Core_Datadriven, self).__init__()
        if core_num == 0:
            raise 
        else:
            print('core number is', core_num)
        self.node_num = node_num
        self.cores = nn.Parameter(torch.randn(core_num, d_core))
        nn.init.xavier_uniform_(self.cores)
        self.query = nn.Conv2d(d_in, d_core, kernel_size=(1, 1))
        self.value = nn.Conv2d(d_in, d_core, kernel_size=(1, 1))
        self.ffn = nn.Sequential(
            nn.Conv2d(d_in + d_core, 4*(d_in + d_core), kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(4*(d_in + d_core), d_out, kernel_size=(1, 1)),
        )
        
        self.d_core = d_core
        self.core_num = core_num

        self.nndropout = nn.Dropout(nndropout)
        self.dropout = dropout
        self.ln = nn.LayerNorm(d_out)

    def forward(self, input, adj=None, *args, **kwargs): 
        # input: (b, f, t, n)
        q = self.query(input) # (b, f, t, n)
        affiliation = torch.einsum('bftn, cf -> bfcn', q, self.cores) / self.d_core**0.5 # (b, f, c, n)
        affiliation_node_to_core = torch.softmax(affiliation, dim=-1) # (b, f, c, n)
        affiliation_core_to_node = torch.softmax(affiliation, dim=-2) # (b, f, c, n)

        v = self.value(input)
        v = torch.einsum('bftn, bfcn -> bftc', v, affiliation_node_to_core)
        v = torch.einsum('bftc, bfcn -> bftn', v, affiliation_core_to_node)
        output = torch.cat([input-v, v], dim=1)
        output = self.ffn(output)
        return output
    


class Core_Decoder(nn.Module):
    def __init__(self, d_in, d_core, d_out, node_num, core_num, nndropout=0.3, dropout=0.08):
        super(Core_Decoder, self).__init__()
        if core_num == 0:
            raise print('Need Core!')
        else:
            print('core number is', core_num)
        self.node_num = node_num
        self.adpative = nn.Parameter(torch.randn(d_core, node_num))
        self.cores_ada = nn.Parameter(torch.randn(core_num, d_core))
        self.cores_dad = nn.Parameter(torch.randn(core_num, d_core))
        nn.init.xavier_uniform_(self.adpative)
        nn.init.xavier_uniform_(self.cores_ada)
        nn.init.xavier_uniform_(self.cores_dad)

        self.query = nn.Conv2d(d_in, d_core, kernel_size=(1, 1))
        self.value_ada = nn.Conv2d(d_in, d_core, kernel_size=(1, 1))
        self.value_dad = nn.Conv2d(d_in, d_core, kernel_size=(1, 1))
        self.ffn = nn.Sequential(
            nn.Conv2d(d_in + d_core, 4*(d_in + d_core), kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(4*(d_in + d_core), d_out, kernel_size=(1, 1)),
        )
        
        self.d_core = d_core
        self.core_num = core_num
        self.tradeoff = nn.Parameter(torch.randn(2))
        nn.init.xavier_uniform_(self.tradeoff)

        self.nndropout = nn.Dropout(nndropout)
        self.dropout = dropout
        self.norm = nn.BatchNorm2d(d_out)

    def forward(self, input, *args, **kwargs): # input: (b, f, t, n)
        
        affiliation_ada = self.cores_ada @ self.adpative / self.d_core**0.5 # (c, n)
        affiliation_ada = torch.softmax(affiliation_ada, dim=-2) # (c, n)
        v1 = self.value_ada(input)
        v1 = torch.einsum('bftn, cn -> bftc', v1, affiliation_ada)
        v1 = torch.einsum('bftc, cn -> bftn', v1, affiliation_ada)
        v1 = input - v1

        q = self.query(input) # (b, f, t, n)
        affiliation_dad = torch.einsum('cf, bftn -> bfcn', self.cores_dad, q) / self.d_core**0.5 # (b, f, c, n)
        affiliation_dad = torch.softmax(affiliation_dad, dim=-2) # (c, n)
        v2 = self.value_ada(input)
        v2 = torch.einsum('bftn, cn -> bftc', v2, affiliation_dad)
        v2 = torch.einsum('bftc, cn -> bftn', v2, affiliation_dad)
        v2 = input - v2

        output = torch.cat([v1, v2], dim=1)
        output = self.ffn(output)
        return output


class Core_Unique(nn.Module):
    def __init__(self, d_series, d_core, d_out, node_num=None, core_num=None, nndropout=0.3, dropout=0.05):
        super(Core_Unique, self).__init__()


        self.gen1 = nn.Conv2d(d_series, d_series, kernel_size=(1, 1))
        self.gen2 = nn.Conv2d(d_series, d_core, kernel_size=(1, 1))
        self.gen3 = nn.Conv2d(d_series + d_core, 4*d_series, kernel_size=(1, 1))
        self.gen4 = nn.Conv2d(4*d_series, d_out, kernel_size=(1, 1))

        self.gen5 = nn.Conv2d(d_series, d_core, kernel_size=(1, 1))
        # gen6 = nn.Conv2d(d_series, d_series, kernel_size=(1, 1))
        self.d_core = d_core
        
        self.nndropout = nn.Dropout(nndropout)
        self.dropout = dropout
        self.norm = nn.BatchNorm2d(d_out)

    def forward(self, input, adj=None, label=None, *args, **kwargs):
        batch_size, channels, nodes, d_series = input.shape # (b, f, n, t)

        # set FFN
        iinput = F.gelu(self.gen1(input))
        # combined_mean = self.gen2(F.gelu(iinput))#  / self.d_core**0.5
        # q = self.gen5(F.gelu(iinput))
        combined_mean = self.gen2(iinput) #iiput or input #  / self.d_core**0.5
        q = self.gen5(input)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=2) # (b, f, n, t)
            ratio = ratio.transpose(-2, -1) # (b, f, n, t) -> (b, f, t, n)
            ratio = ratio.reshape(-1, nodes) # (b, f, t, n) -> (b*f*t, n)
            indices = torch.multinomial(ratio, int(0.9*nodes)) # (b*f*t, m) m<n
            indices = indices.view(batch_size, channels, d_series, -1).transpose(-2, -1) # -> (b, f, t, m) -> (b, f, m, t)
            combined_mean = torch.gather(combined_mean, 2, indices) # (b, f, n, t) -> (b, f, m, t)
            q = torch.gather(q, 2, indices) # (b, f, n, t) -> (b, f, m, t)
            weight = F.softmax(combined_mean, dim=2) # (b, f, m, t)
            # combined_mean = torch.sum(combined_mean * weight, dim=2, keepdim=True).repeat(1, 1, nodes, 1)
            combined_mean = torch.sum(q * weight, dim=2, keepdim=True).repeat(1, 1, nodes, 1)
        else:
            weight = F.softmax(combined_mean, dim=2) # (b, f, n, t)
            # combined_mean = torch.sum(combined_mean * weight, dim=2, keepdim=True).repeat(1, 1, nodes, 1)
            combined_mean = torch.sum(q * weight, dim=2, keepdim=True).repeat(1, 1, nodes, 1)

        # mlp fusion
        combined_mean_cat = torch.cat([input-combined_mean, combined_mean], 1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        # output = combined_mean_cat

        combined_mean_cat = self.norm(self.nndropout(combined_mean_cat) + input)

        return combined_mean_cat # output