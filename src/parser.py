import argparse

import matplotlib.pyplot as plt
import pandas as pd

from utils import *


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run .")
    """
    正则表达式的规则－参数个数：
    nargs='*' 　　　表示参数可设置零个或多个
    nargs=' '+' 　　表示参数可设置一个或多个
    nargs='?'　　　表示参数可设置零个或一个
    """

    parser.add_argument("--edge-path",
                        nargs="?",
                        default="/home/mrxia/PycharmProjects/ClusterGCN/input/edges.csv",
                        help="Edge list csv.")

    parser.add_argument("--features-path",
                        nargs="?",
                        default="/home/mrxia/PycharmProjects/ClusterGCN/input/features.csv",
                        help="Features csv.")

    parser.add_argument("--target-path",
                        nargs="?",
                        default="/home/mrxia/PycharmProjects/ClusterGCN/input/target.csv",
                        help="Target classes csv.")

    parser.add_argument("--clustering-method",
                        nargs="?",
                        default="metis",
                        help="Clustering method for graph decomposition.")

    parser.add_argument("--epochs",
                        type=int,
                        default=200,
                        help="Number of training epochs.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for train-test split")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.2,
                        help="Dropout parameter.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
                        help="Learning rate.")

    parser.add_argument("--test-ratio",
                        type=float,
                        default=0.9,
                        help="Test data ratio.")

    parser.add_argument("--cluster-number",
                        type=int,
                        default=20,
                        help="Number of clustering extracted.")

    parser.add_argument("--layers",
                        type=list,
                        default=[512, 512, 512],
                        help="Number of each layer hidden.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parameter_parser()
    tab_printer(args)

