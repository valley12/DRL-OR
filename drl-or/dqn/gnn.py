import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv

demands_list = [8, 32, 64]

hparams = {
    'l2': 0.1,
    'dropout_rate': 0.01,
    'link_state_dim': 20,
    'readout_units': 35,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'T': 4,
    'num_demands': len(demands_list)
}

class GN(nn.Module):
    pass

# Message passing for Edge Feature
class MPNN(nn.Module):

    def __init__(self, link_state_dim=20, readout_units=35, dropout_rate=0.01, message_steps=4, act_dim=5):

        super(MPNN, self).__init__()
        # 消息传递次数
        self.message_steps = message_steps

        # 消息函数
        self.Message = nn.Sequential(
            nn.Linear(link_state_dim * 2, link_state_dim),
            nn.SELU()
        )

        # 更新函数
        self.Update = nn.GRUCell(link_state_dim, link_state_dim)

        # 读出函数,输出选择的动作值
        self.Readout = nn.Sequential(
            nn.Linear(link_state_dim, readout_units),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(readout_units, readout_units),
            nn.SELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(readout_units, act_dim)
        )
        # L2 regularization is added through weight decay in the optimizer

    def forward(self, link_state, main_edge_index, neigh_edge_index, num_edges):
        # 对边的信息传递N次
        for _ in range(self.message_steps):
            # We have the combination of the hidden states of the main edges with the neighbours
            main_edges_state = link_state[main_edge_index]
            neigh_edges_state = link_state[neigh_edge_index]

            edges_concat = torch.cat([main_edges_state, neigh_edges_state], dim=1)

            ### 1.a Message passing for link with all it's neighbours
            outputs = self.Message(edges_concat)

            ### 1.b Sum of output values according to link id index
            edges_inputs = torch.zeros(num_edges.max().item(), outputs.size(1), device=outputs.device)
            edges_inputs = edges_inputs.index_add(0, neigh_edge_index, outputs)

            ### 2. Update for each link
            # GRUCell needs a 2D tensor as input, update with GRUCell
            link_state = self.Update(edges_inputs, link_state)

        edges_combi_outputs = torch.sum(link_state, dim=0)

        r = self.Readout(edges_combi_outputs)

        return r


# GCN + GAT + Dueling DQN
class GCNVANet(nn.Module):
    def __init__(self, state_dim=20, action_dim=5):

        super().__init__()

        # GCN Convolution
        self.gcn_conv = GCNConv(in_channels=state_dim, out_channels=state_dim)

        # GAT Convolution, single head
        self.gat_conv = GATConv(in_channels=state_dim, out_channels=state_dim, heads=1)

        self.action_readout = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, action_dim)
        )

        self.value_readout = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        # 图卷积网络处理节点特征
        x = F.relu(self.gcn_conv(x))
        x = F.relu(self.gat_conv(x))

        x = torch.sum(x, dim=0)

        A = self.action_readout(x)
        V = self.value_readout(x)

        Q = V + A - A.mean(1).view(-1, 1)
        return Q

class GCNQNet(nn.Module):
    def __init__(self, state_dim=20, action_dim=1):

        super().__init__()

        # GCN Convolution
        self.gcn_conv = GCNConv(in_channels=state_dim, out_channels=state_dim)

        # GAT Convolution, single head
        # self.gat_conv = GATConv(in_channels=state_dim, out_channels=state_dim, heads=1)

        self.gru = nn.GRUCell(input_size=state_dim, hidden_size=state_dim)

        self.lin = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, action_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # 图卷积网络处理节点特征
        node_feat = F.relu(self.gcn_conv(x, edge_index=edge_index))
        node_feat = self.gru(node_feat, x)
        node_feat = torch.sum(node_feat, dim=1)
        Q = self.lin(node_feat)
        return Q

