import torch
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
from texttable import Texttable
from scipy.sparse import coo_matrix

from sklearn import manifold, datasets
from visdom import Visdom

def tab_printer(args):
    """参数表格方法
+-------------------+----------------------+
|     Parameter     |        Value         |
+===================+======================+
| Cluster number    | 10                   |
+-------------------+----------------------+
| Clustering method | metis                |
+-------------------+----------------------+
| Dropout           | 0.500                |
+-------------------+----------------------+
| Edge path         | ./input/edges.csv    |
+-------------------+----------------------+
| Epochs            | 200                  |
+-------------------+----------------------+
| Features path     | ./input/features.csv |
+-------------------+----------------------+
| Layers            | [16, 16, 16]         |
+-------------------+----------------------+
| Learning rate     | 0.010                |
+-------------------+----------------------+
| Seed              | 42                   |
+-------------------+----------------------+
| Target path       | ./input/target.csv   |
+-------------------+----------------------+
| Test ratio        | 0.900                |
+-------------------+----------------------+
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    # 将字符串的第一个字母变成大写,其他字母变小写

    print(t.draw())


def graph_reader(path):
    """读取边表，构图"""
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    return graph


def feature_reader(path):
    """读取每个节点的特征"""
    features = pd.read_csv(path)
    node_index = features["node_id"].values.tolist()
    feature_index = features["feature_id"].values.tolist()
    feature_values = features["value"].values.tolist()
    node_count = max(node_index) + 1
    feature_count = max(feature_index) + 1
    features = coo_matrix((feature_values, (node_index, feature_index)), shape=(node_count, feature_count)).toarray()

    return features


def target_reader(path):
    """读取各个节点的标签"""
    target = np.array(pd.read_csv(path)["target"]).reshape(-1, 1)
    return target


def visualizetion(features, target):
    #tsne = manifold.TSNE(n_components=3,
    #                     init='pca',
    #                     random_state=0)
    #result = tsne.fit_transform(features)
    result = features
    target.squeeze()

    vis = Visdom()
    vis.scatter(
        X=result,
        Y=target+1,
        opts=dict(markersize=5, title='Dimension reduction to %dD' % (result.shape[1])),
    )


def visualizetion_batchs_Graph(clustering_machine):
    colors = {
        0: "yellow",
        1: "red",
        2: "blue",
    }
    for batch in range(clustering_machine.batch_num):
        edge_index_np = clustering_machine.batch_edges[batch].cpu().numpy()
        num_of_nodes = clustering_machine.batch_nodes[batch].shape[0]
        target_nodes = clustering_machine.batch_targets[batch].cpu().numpy().squeeze()
        edge_index_tuples = list(zip(edge_index_np[0, :], edge_index_np[1, :]))

        ig_graph = ig.Graph()
        ig_graph.add_vertices(num_of_nodes)
        ig_graph.add_edges(edge_index_tuples)

        visual_style = {}
        visual_style['layout'] = ig_graph.layout_kamada_kawai()
        visual_style['vertex_color'] = [colors[target] for target in target_nodes]

        print('Plotting results ... (it may take couple of seconds).')
        ig.plot(ig_graph, **visual_style)













