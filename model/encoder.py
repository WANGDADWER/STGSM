import os
import pickle

import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import RGCNConv, GCNConv, HypergraphConv

from utils.constants import *
from utils.graph import load_static_graph, load_hypergraph, load_hetero_graph
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.2):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):


        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = nn.functional.softmax(self.linear2(torch.nn.functional.gelu(self.linear1(emb))), dim=0)

        out = torch.sum(emb_score * emb, dim=0)
        return out


class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.1, opt=None):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)


        self.ntoken = ntoken

        self.ninp = ninp

        self.hypergcn = HypergraphConv(in_channels=self.ninp, out_channels=self.ninp)

        self.lightgcn_layers = int(getattr(opt, 'lightgcn_layers', 2))
        self.lightgcn_gamma = float(getattr(opt, 'lightgcn_gamma', 0.1))
        if self.lightgcn_layers < 1:
            raise ValueError('lightgcn_layers must be at least 1')
        if not 0.0 <= self.lightgcn_gamma <= 1.0:
            raise ValueError('lightgcn_gamma must be in [0, 1]')
        self.global_hypergcn = HypergraphConv(in_channels=self.ninp, out_channels=self.ninp)

        self.att = nn.Parameter(torch.zeros(1, self.ninp))
        self.att_m = nn.Parameter(torch.zeros(self.ninp, self.ninp))
        self.dropout = nn.Dropout(dropout)


        self.n_channel = 3
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.ninp, self.ninp)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.ninp)) for _ in range(self.n_channel)])
        self.fus = Fusion(self.ninp)

        self.register_buffer("Graph", None, persistent=False)
        self.gamma = 0.1
        self.gamma_v1 = nn.Parameter(torch.tensor(0.1))
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)
        init.xavier_normal_(self.weights[0])
        init.xavier_normal_(self.weights[1])
        init.xavier_normal_(self.att)
        init.xavier_normal_(self.att_m)

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))

    def get_normalize_graph(self, index):
        index = index.cpu()
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse.IntTensor(index, data,
                                            torch.Size([self.ntoken, self.ntoken]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero()
        data = dense[dense >= 1e-9]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.ntoken, self.ntoken]))
        Graph = Graph.coalesce().to_dense().to(self.embedding.weight.device)
        return Graph


    def lightgcn(self, all_emb):


        embs = [all_emb]


        for layer in range(self.lightgcn_layers):
            norm = torch.norm(all_emb, dim=1) + 1e-12
            all_emb = all_emb / norm[:, None]
            all_emb = torch.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs_zero = embs[0]
        embs_prop = torch.mean(torch.stack(embs[1:], dim=1), dim=1)
        light_out = (self.lightgcn_gamma * embs_zero) + ((1 - self.lightgcn_gamma) * embs_prop)
        return light_out


    def diff_loss(self, shared_embedding, task_embedding):
        shared_embedding = shared_embedding -  torch.mean(shared_embedding, 0)
        task_embedding = task_embedding - torch.mean(task_embedding, 0)


        shared_embedding = nn.functional.normalize(shared_embedding, dim=1, p=2)
        task_embedding = nn.functional.normalize(task_embedding, dim=1, p=2)

        correlation_matrix = task_embedding.t() @ shared_embedding
        loss_diff = torch.mean(torch.square_(correlation_matrix)) * 0.01
        loss_diff = torch.where(loss_diff > 0, loss_diff, 0)
        return loss_diff

    def forward(self, heter_graph, hyper_graph, hyper_graph_micro):
        heter_graph_edge_index = heter_graph.edge_index


        u_emb = self.self_gating(self.embedding.weight, 0)
        c_emb = self.self_gating(self.embedding.weight, 1)

        if self.Graph is None:
            self.Graph = self.get_normalize_graph(heter_graph_edge_index)

        u_emb = self.lightgcn(u_emb)

        hg_embeddings = []
        for i in range(len(hyper_graph)):
            subhg_embedding = self.hypergcn(c_emb, hyper_graph[i].edge_index.to(c_emb.device))
            if i == 0:
                hg_embeddings.append(subhg_embedding)
            else:
                subhg_embedding = self.fus(hg_embeddings[-1], subhg_embedding)
                hg_embeddings.append(subhg_embedding)

        h_emb = hg_embeddings[-1]

        hg_embeddings_micro = []
        for i in range(len(hyper_graph_micro)):
            subhg_embedding = self.hypergcn(c_emb, hyper_graph_micro[i].edge_index.to(c_emb.device))
            if i == 0:
                hg_embeddings_micro.append(subhg_embedding)
            else:
                subhg_embedding = self.fus(hg_embeddings_micro[-1], subhg_embedding)
                hg_embeddings_micro.append(subhg_embedding)

        h_emb_micro = hg_embeddings_micro[-1]

        high_embs, attention_score = self.channel_attention(u_emb, h_emb, h_emb_micro)

        return high_embs, 0.0

    def channel_attention(self, *channel_embeddings):
        weights = []

        for embedding in channel_embeddings:
            weights.append(
                torch.sum(torch.multiply(self.att, torch.matmul(embedding, self.att_m)), 1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim=-1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score


class GraphEncoder(nn.Module):
    def __init__(self, opt, Type, hypergraph_list, hypergraph_list_micro, dropout=0.15):
        super(GraphEncoder, self).__init__()
        self.dropedge=opt.dropout

        self.ntoken = opt.ntoken
        self.user_size = opt.user_size
        self.ninp = opt.d_word_vec
        self.output_dim = opt.d_word_vec

        self.heter_graph = load_static_graph(opt.data_path,Type)
        self.hyper_graph = hypergraph_list
        self.hyper_graph_micro = hypergraph_list_micro

        self.global_hyper_graph = load_hypergraph(opt.data_path,Type)
        self.gnn_layer = GraphNN(self.ntoken, self.ninp, opt=opt)


    def forward(self,input, input_timestamp,train=True):

        batch_size, max_len = input.size()
        user_embedding_lookup, diff_loss = self.gnn_layer(self.heter_graph, self.hyper_graph, self.hyper_graph_micro)


        self.last_user_embedding_lookup = user_embedding_lookup
        user_input=input.contiguous().view(batch_size*max_len,1)
        user_social_embedding_one_hot=torch.zeros(batch_size*max_len, self.ntoken, device=input.device)
        user_social_embedding_one_hot=user_social_embedding_one_hot.scatter_(1, user_input, 1)

        user_embedding=torch.einsum("bt,td->bd",user_social_embedding_one_hot,user_embedding_lookup).view(batch_size,max_len,
                                                                                                          self.ninp)
        return user_embedding

    def lookup_encoded_users(self, ids):


        table = getattr(self, 'last_user_embedding_lookup', None)
        if table is None:
            raise RuntimeError(
                'encoded user lookup requested before the encoder forward pass; the propagated '
                'table is published as last_user_embedding_lookup during forward()')
        return table(ids) if callable(table) else table[ids]
