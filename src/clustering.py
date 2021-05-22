import metis
import torch
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import scipy.sparse as sp


class ClusteringMachine(object):
    def __init__(self, args, graph, features, target):
        """
        :param args: Arguments object with parameters.
        :param graph: Networks Graph.
        :param features: Features matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self._set_sizes()

    def _set_sizes(self):
        """
        Setting the features and class count.
        """
        self.features_count = self.features.shape[1]
        self.target_count = np.max(self.target) + 1

    def decompose(self):
        """
        Decompose the graph, partitioning the features and target, creating Torch arrays.
        """
        if self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            self.metis_clustering()
        else:
            print("\nRandom graph clustering started.\n")
            self.random_clustering()

        self.general_data_partitioning()
        self.transfer_edges_and_nodes()
        self.general_batch_partitioning()
        self.get_normalize_mx()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """
        (st, parts) = metis.part_graph(self.graph, self.args.cluster_number)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def get_normalize_mx(self):
        self.norm_mx = {}
        for batch in range(self.batch_num):
            node_size = self.batch_nodes[batch].shape[0]
            num_edges = self.batch_edges[batch].shape[1]
            values = torch.from_numpy(np.array([1] * num_edges).squeeze())
            identiy = sp.eye(node_size).tocoo().astype(np.float32)
            indices = torch.from_numpy(
                np.vstack((identiy.row, identiy.col)).astype(np.int64)
            )
            identiy_values = torch.from_numpy(identiy.data)
            shape = torch.Size(identiy.shape)
            identiy = torch.sparse.FloatTensor(indices, identiy_values, shape)

            A = torch.sparse.FloatTensor(self.batch_edges[batch], values, shape)
            A = A + identiy

            rowsum = A.to_dense().sum(1)
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = torch.diag(r_inv)
            mx = torch.smm(A.t(), r_mat_inv.t()).t()

            self.norm_mx[batch] = mx

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test-splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}

        for cluster in self.clusters:
            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes())
                                            if self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]

            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}  # id_node: subgraph_idx

            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]]for edge in subgraph.edges()] + \
                                     [[mapper[edge[1]], mapper[edge[0]]]for edge in subgraph.edges()]

            self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()),
                                                                                         test_size=self.args.test_ratio)

            self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])

            self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])

            self.sg_features[cluster] = self.features[self.sg_nodes[cluster], :]

            self.sg_targets[cluster] = self.target[self.sg_nodes[cluster], :]

    def transfer_edges_and_nodes(self):
        """
        Transfer the data to PyTorch format.
        """
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])

    def general_batch_partitioning(self):
        random.shuffle(self.clusters)
        self.batch_nodes = {}
        self.batch_edges = {}
        self.batch_train_nodes = {}
        self.batch_test_nodes = {}
        self.batch_features = {}
        self.batch_targets = {}

        self.batch_num = 0
        for i in range(0, len(self.clusters), 2):
            if i != len(self.clusters) - 1:
                cluster1, cluster2 = self.clusters[i], self.clusters[i+1]
                subgraph = self.graph.subgraph(
                    [node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster1
                     or self.cluster_membership[node] == cluster2]
                )
            else:
                cluster = self.clusters[i]
                subgraph = self.graph.subgraph(
                    [node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster]
                )

            self.batch_nodes[self.batch_num] = [node for node in sorted(subgraph.nodes())]

            mapper = {node: i for i, node in enumerate(sorted(self.batch_nodes[self.batch_num]))}

            self.batch_edges[self.batch_num] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] + \
                                               [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]

            self.batch_train_nodes[self.batch_num], self.batch_test_nodes[self.batch_num] = \
                train_test_split(list(mapper.values()), test_size=self.args.test_ratio)

            self.batch_features[self.batch_num] = self.features[self.batch_nodes[self.batch_num], :]

            self.batch_targets[self.batch_num] = self.target[self.batch_nodes[self.batch_num], :]

            self.batch_nodes[self.batch_num] = torch.LongTensor(self.batch_nodes[self.batch_num])
            self.batch_edges[self.batch_num] = torch.LongTensor(self.batch_edges[self.batch_num]).t()
            self.batch_train_nodes[self.batch_num] = torch.LongTensor(self.batch_train_nodes[self.batch_num])
            self.batch_test_nodes[self.batch_num] = torch.LongTensor(self.batch_test_nodes[self.batch_num])
            self.batch_targets[self.batch_num] = torch.LongTensor(self.batch_targets[self.batch_num])
            self.batch_features[self.batch_num] = torch.FloatTensor(self.batch_features[self.batch_num])

            self.batch_num += 1






