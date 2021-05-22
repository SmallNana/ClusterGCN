import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import ClusterGCNConv
import scipy.sparse as sp


class StackedGCN(torch.nn.Module):
    """
    Mulit-layer GCN model.
    """
    def __init__(self, args, input_channels, output_channels):
        """
        :param args: Arguments objects
        :param input_channels: Number of features
        :param output_channels: Number of target features
        """
        super(StackedGCN, self).__init__()
        self.args = args
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers based on args
        """
        self.layers = []
        self.args.layers = [self.input_channels] + self.args.layers + [self.output_channels]

        for i, _ in enumerate(self.args.layers[:-1]):
            # self.layers.append(ClusterGCNConv(self.args.layers[i], self.args.layers[i+1]))
            self.layers.append(GraphConvolution(self.args.layers[i], self.args.layers[i+1]))

        self.layers = ListModule(*self.layers)

    def forward(self, edges, features):
        """
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :param features: Feature matrix input FLoatTensor.
        :return predictions: Prediction matrix output FLoatTensor.
        """
        for i, _ in enumerate(self.args.layers[:-2]):
            features = F.relu(self.layers[i](features, edges))
            if i > 1:
                features = F.dropout(features, p=self.args.dropout, training=self.training)

        features = self.layers[i + 1](features, edges)
        predictions = F.log_softmax(features, dim=1)

        return predictions


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)

        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, diag_lambda=0, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diag_lambda = diag_lambda
        self.weight = torch.nn.Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)  # tensor从均匀分布中抽样数值进行填充
        if self.bias is not None:
            self.bias = torch.nn.init.zeros_(self.bias)

    def get_normalize_mx(self, input, adj):
        node_size = input.shape[0]
        num_edges = adj.shape[1]
        values = torch.from_numpy(np.array([1]*num_edges).squeeze())
        identiy = sp.eye(node_size).tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((identiy.row, identiy.col)).astype(np.int64)
        )
        identiy_values = torch.from_numpy(identiy.data)
        shape = torch.Size(identiy.shape)

        identiy = torch.sparse.FloatTensor(indices, identiy_values, shape)

        A = torch.sparse.FloatTensor(adj, values, shape)

        A = A + identiy

        rowsum = A.to_dense().sum(1)
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.smm(A.t(), r_mat_inv.t()).t()

        return mx

    def forward(self, input, adj):
        """
        input = input.cpu()
        adj = adj.cpu()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mx = self.get_normalize_mx(input, adj).to(device)
        input = input.to(device)
        """
        support = torch.mm(input, self.weight)
        A = self.diag_lambda * torch.diag(adj.to_dense()) + adj.to_dense()
        output = torch.spmm(A, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output



