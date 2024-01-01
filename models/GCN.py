import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
dtype = torch.float

from helper.config import setup_seed
setup_seed(2021)

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1) 
    AD = np.dot(A, Dn)
    return AD


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(GraphConvolution, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.zeros((input_size, output_size), device=torch.device('cuda', 0), dtype=dtype), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_size, device=torch.device('cuda', 0), dtype=dtype), requires_grad=True)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, A):
        x = torch.einsum("ijk, kl->ijl", [x, self.weight])
        x = torch.einsum("ij, kjl->kil", [A, x])
        x = x + self.bias

        return x


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_size, hidden_size, device)
        self.gcn2 = GraphConvolution(hidden_size, output_size, device)

    def forward(self, x, A):
        x = self.gcn1(x, A)
        x = F.relu(x)
        x = self.gcn2(x, A)
        x = F.relu(x)
        return x


class GCN_Precodition(nn.Module):
    def __init__(self, len, map_width, patch_num, embed_dim, in_chans, Beta):#, input_size, feature_size, hidden_size, device):
        super(GCN_Precodition, self).__init__()
        self.patch_num = patch_num
        self.embed_dim = embed_dim
        self.map_width = map_width
        self.Beta = Beta
        self.in_chans = in_chans
        self.input_size = 2
        self.len = len
        self.device = torch.device('cuda', 0)
        self.gcn = GCN(input_size=2, hidden_size=512, output_size=128, device=torch.cuda)
        self.maxpool = nn.MaxPool2d(2,stride=2)
        
    def forward(self, x, A, hidden=None):
        A = np.sum(A.cpu().detach().numpy(), axis=0)
        A = np.sum(A, axis=0)
       
        A = A + np.eye(A.shape[0])*np.max(A)
        A = np.where(np.abs(A) < np.abs(np.mean(A))*(1 + self.Beta), 0, A) 
        A = torch.tensor(normalize_digraph(A), device=self.device, dtype=dtype)
 
        enc_gcn = None
        x = self.maxpool(x).reshape(-1, self.len, 2, int(self.map_width/2), int(self.map_width/2))

        for i in range(x.shape[1]):
            if i == 0:
                enc_gcn = self.gcn(x[:, i].permute(0, 2, 3, 1).reshape(-1, self.patch_num, 2), A).reshape(-1, 1, self.patch_num, self.embed_dim)
            else:
                enc_gcn = torch.cat((self.gcn(x[:, i].permute(0, 2, 3, 1).reshape(-1, self.patch_num, 2), A).reshape(-1, 1, self.patch_num, self.embed_dim), enc_gcn), dim=1)
        return enc_gcn

    def init_hidden(self, x):
        return torch.zeros((2, x.size(0), self.hidden_size), device=self.device, dtype=dtype)
