import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import config


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GlobalTrajPredictor(nn.Module):

    def __init__(self, device):

        super(GlobalTrajPredictor, self).__init__()

        self.device = device

        # STGCN
        in_channels = 3
        edge_importance_weighting = True

        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 5
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 128, kernel_size, 1),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 256, kernel_size, 1),
            st_gcn(256, 256, kernel_size, 1),
            st_gcn(256, 256, kernel_size, 1),
        ))
        self.num_features = 256
        self.mlp = nn.Linear(256 * 24, self.num_features)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # Iterative regression
        self.n_iter = 3
        mean_rottrans = np.load(config.rottrans_mean)
        init_rottrans = torch.from_numpy(mean_rottrans).unsqueeze(0).float()
        self.register_buffer('init_rottrans', init_rottrans)

        self.fc1 = nn.Linear(self.num_features + 3, self.num_features)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(self.num_features, self.num_features)
        self.drop2 = nn.Dropout()
        self.dec = nn.Linear(self.num_features, 3)
        nn.init.xavier_uniform_(self.dec.weight, gain=0.01)

        self.mlp1_rot = nn.Linear(self.num_features * config.seqlen, self.num_features)
        self.relu1_rot = nn.ReLU()
        self.mlp2_rot = nn.Linear(self.num_features, self.num_features // 2)
        self.relu2_rot = nn.ReLU()
        self.mlp3_rot = nn.Linear(self.num_features // 2, 2)

    def forward(self, x_joint):

        N, T, V, C = x_joint.size()

        # STGCN
        x = x_joint.clone().permute(0, 2, 3, 1).contiguous()  # N, V, C, T
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()  # N, C, T, V
        x = x.view(N, C, T, V)

        for gcn_layer, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
            x, _ = gcn(x, self.A * importance)
        x = x.permute(0, 2, 3, 1).contiguous()  # N, T, V, c
        x = x.reshape(N, T, -1)
        x = self.mlp(x)

        y_rot = self.mlp1_rot(x.clone().reshape(N, -1))
        y_rot = self.relu1_rot(y_rot)
        y_rot = self.mlp2_rot(y_rot)
        y_rot = self.relu2_rot(y_rot)
        pred_rot = self.mlp3_rot(y_rot)
        rot_x = (torch.tanh(pred_rot[:, :1]) * 30 / 180 * np.pi).expand(-1, config.seqlen).reshape(-1)
        rot_y = (torch.tanh(pred_rot[:, 1:2]) * 30 / 180 * np.pi).expand(-1, config.seqlen).reshape(-1)

        cosx = torch.cos(rot_x)
        sinx = torch.sin(rot_x)
        zeros = cosx.detach() * 0
        ones = zeros.detach() + 1
        rotmat_x = torch.stack([ones, zeros, zeros,
                                zeros, cosx, -sinx,
                                zeros, sinx, cosx], dim=1).reshape(-1, 3, 3)

        cosy = torch.cos(rot_y)
        siny = torch.sin(rot_y)
        zeros = cosy.detach() * 0
        ones = zeros.detach() + 1
        rotmat_y = torch.stack([cosy, zeros, siny,
                                zeros, ones, zeros,
                                -siny, zeros, cosy], dim=1).reshape(-1, 3, 3)

        R = rotmat_y @ rotmat_x

        x = (x_joint.reshape(-1, 24, 3) @ R).reshape(N, T, V, C)
        x = x.permute(0, 2, 3, 1).contiguous()  # N, V, C, T
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()  # N, C, T, V
        x = x.view(N, C, T, V)

        for gcn_layer, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
            x, _ = gcn(x, self.A * importance)
        x = x.permute(0, 2, 3, 1).contiguous()  # N, T, V, c
        x = x.reshape(N, T, -1)
        x = self.mlp(x)

        y_trans = x.clone().reshape(N * T, -1)
        # iterative regression
        init_trans = self.init_rottrans[:, 2:].expand(N * T, -1)
        pred_trans = init_trans
        for i in range(self.n_iter):
            yc = torch.cat([y_trans, pred_trans], 1)
            yc = self.fc1(yc)
            yc = self.drop1(yc)
            yc = self.fc2(yc)
            yc = self.drop2(yc)
            pred_trans = self.dec(yc) + pred_trans

        return rot_x, rot_y, R, pred_trans

    def forward_T(self, delta_trans, seqlen):

        T = delta_trans.reshape([-1, seqlen, 3])
        T[:, 0, :2] = 0
        T[:, :, :2] = torch.cumsum(T[:, :, :2], dim=1)

        return T

    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.01)


class Graph():
    """ The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(self,
                 strategy='spatial',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        # get edge
        self.num_node = 24
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(17, 19), (19, 21), (21, 23),
                         (16, 18), (18, 20), (20, 22),
                         (12, 13), (12, 14), (12, 15),
                         (3, 6), (6, 9), (9, 13), (9, 14),  # upper body
                         (2, 5), (5, 8), (8, 11),  # right leg
                         (1, 4), (4, 7), (7, 10),  # left leg
                         (0, 1), (0, 2), (0, 3)]
        self.edge = self_link + neighbor_link
        self.center = 0

        # get adjacency matrix
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                    center] > self.hop_dis[i, self.
                                    center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        # print(x.shape)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        # print(x.shape)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        # print(x.shape)
        return x.contiguous(), A


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.5,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
