import torch
from parser import parameter_parser
from clustering import ClusteringMachine
from clustergcn import ClusterGCNTrainer
from utils import *


def main():
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)

    graph = graph_reader(args.edge_path)
    features = feature_reader(args.features_path)
    target = target_reader(args.target_path)

    clustering_machine = ClusteringMachine(args, graph, features, target)
    clustering_machine.decompose()
    """
    生成每个batch的降维图
    for batch in range(clustering_machine.batch_num):
        visualizetion(graph, clustering_machine.batch_features[batch], clustering_machine.batch_targets[batch])
    """

    gcn_trainer = ClusterGCNTrainer(args, clustering_machine)
    gcn_trainer.train()
    gcn_trainer.test()



if __name__ == "__main__":
    main()
