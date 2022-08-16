import argparse
import sys
import os
import random
import numpy as np

import gencat
from utils_gencat import config_diagonal, feature_extraction
from converter import convert_to_planetoid

sys.path.append("./")

datasets_to_convert = [
    "cornell",
    "texas",
    "wisconsin",
    'blogcatalog',
    'wiki',
    'flickr',
    'actor',
    'chameleon',
    'squirrel']


def main(args):
    from utils import load_data, save_graph_2, get_path_to_top_dir
    from models.dataset_utils import DataLoader
    _ = DataLoader(args.dataset, data_dir="./data/")  # download dataset if not exist

    if args.dataset in datasets_to_convert:
        convert_to_planetoid(args.dataset)
    adj, features, labels = load_data(args.dataset)
    print(type(adj))  # <class 'scipy.sparse.csr.csr_matrix'>
    print(adj.shape)  # (n_data, n_data)
    print(type(features))  # <class 'numpy.ndarray'>
    print(features.shape)  # (n_data, n_feature)
    print(type(labels))  # <class 'list'>
    print(len(labels))  # n_data

    num_classes = len(set(labels))
    print("class:", num_classes)

    top_dir = get_path_to_top_dir()
    n_iter = args.n_iter

    M, D, class_size, H, node_degree = feature_extraction(adj, features, labels)

    if args.exp == "hetero_homo":
        k = num_classes
        M_diag = 0  # initialization
        for i in range(k):
            M_diag += M[i, i]
        M_diag /= k  # calculate average of intra-class connections
        # Integer representation of the percentage of the original data to be adjusted to 10 steps
        base_value = int(M_diag * 10)

        # Generate parameters according to the percentage of in-class connections
        # in the original data
        params = list(range(base_value - 9, base_value + 1))
        print(params)

        data_dir = f"{top_dir}/data/GenCAT_Exp_{args.exp}/"
        os.makedirs(data_dir, exist_ok=True)

        for x_ in params:
            for i in range(n_iter):
                M_config, D_config = config_diagonal(M, D, x_)
                adj, features, labels = gencat.gencat(
                    M_config, D_config, H, class_size=class_size, theta=node_degree)
                print(type(adj))  # <class 'scipy.sparse.dok.dok_matrix'>
                print(adj.shape)  # (n_data, n_data)
                data_name = f"GenCAT_{args.dataset}_{x_}_{i}"

                save_graph_2(
                    adj,
                    features,
                    labels,
                    dataset_str=data_name,
                    data_dir=data_dir)

    elif args.exp == "attribute":
        data_dir = f"{top_dir}/data/GenCAT_Exp_{args.exp}/"
        os.makedirs(data_dir, exist_ok=True)
        params = [0] + [2**i for i in range(8)] + ['rand']  # 0~128, random
        for alpha in params:
            for i in range(n_iter):
                if alpha != 'rand':
                    H_config = (H + alpha * np.average(H)) / (alpha + 1)
                else:
                    H_config = np.zeros(H.shape) + np.average(H)

                adj, features, labels = gencat.gencat(
                    M, D, H=H_config, class_size=class_size, theta=node_degree)
                print(type(adj))  # <class 'scipy.sparse.dok.dok_matrix'>
                print(adj.shape)  # (n_data, n_data)
                data_name = f"GenCAT_{args.dataset}_{alpha}_{i}"

                save_graph_2(
                    adj,
                    features,
                    labels,
                    dataset_str=data_name,
                    data_dir=data_dir)

    elif args.exp == "scalability_edge":
        # parameters to be changed: n(node), m(edge)
        data_dir = f"{top_dir}/data/GenCAT_Exp_scalability_edge/"
        os.makedirs(data_dir, exist_ok=True)
        params = [5000, 10000, 15000, 20000, 25000]  # change edge
        n = len(node_degree)
        for m in params:
            for i in range(n_iter):
                adj, features, labels = gencat.gencat(
                    M, D, H, class_size=class_size, n=n, m=m)
                print(type(adj))  # <class 'scipy.sparse.dok.dok_matrix'>
                print(adj.shape)  # (n_data, n_data)
                print(adj.nnz)  # number of edges
                data_name = f"GenCAT_{args.dataset}_{m}_{i}"

                save_graph_2(
                    adj,
                    features,
                    labels,
                    dataset_str=data_name,
                    data_dir=data_dir)

    elif args.exp == "scalability_node_edge":
        # parameters to be changed: n(node), m(edge)
        data_dir = f"{top_dir}/data/GenCAT_Exp_scalability_node_edge/"
        os.makedirs(data_dir, exist_ok=True)
        node_params = [3000, 6000, 9000, 12000, 15000, 60000, 120000]
        edge_params = [5000, 10000, 15000, 20000, 25000, 100000, 200000]

        for i in range(len(node_params)):
            for j in range(n_iter):
                adj, features, labels = gencat.gencat(
                    M, D, H, class_size=class_size, n=node_params[i], m=edge_params[i])
                print(type(adj))  # <class 'scipy.sparse.dok.dok_matrix'>
                print(adj.shape)  # (n_data, n_data)
                print(adj.nnz)  # number of edges
                data_name = f"GenCAT_{args.dataset}_{node_params[i]}_{edge_params[i]}_{j}"

                save_graph_2(
                    adj,
                    features,
                    labels,
                    dataset_str=data_name,
                    data_dir=data_dir)

    elif args.exp == "classsize":
        data_dir = f"{top_dir}/data/GenCAT_Exp_{args.exp}/"
        os.makedirs(data_dir, exist_ok=True)

        def get_class_size(alpha):
            if alpha == "flat":
                # flat pattern
                class_size = [1 / num_classes] * num_classes
            else:
                tmp = 1
                class_size = list()
                for _ in range(num_classes - 1):
                    class_size.append(tmp * alpha)
                    tmp -= tmp * alpha

                class_size.append(class_size[-1] / alpha * (1 - alpha))
                # sum(class_size) == 1
                # print(np.array(class_size), sum(class_size))
                random.shuffle(class_size)
            return class_size

        alpha_list = [0.4, 0.5, 0.6, 0.7, "flat"]

        for alpha in alpha_list:
            class_size_ = get_class_size(alpha)

            for i in range(n_iter):
                adj, features, labels = gencat.gencat(
                    M, D, H, class_size=class_size_, theta=node_degree)
                print(type(adj))  # <class 'scipy.sparse.dok.dok_matrix'>
                print(adj.shape)  # (n_data, n_data)
                print(adj.nnz)  # number of edges
                data_name = f"GenCAT_{args.dataset}_{alpha}_{i}"

                save_graph_2(
                    adj,
                    features,
                    labels,
                    dataset_str=data_name,
                    data_dir=data_dir)

    else:
        adj, features, labels = gencat.gencat(
            M, D, H, class_size=class_size, theta=node_degree)
        print(type(adj))  # <class 'scipy.sparse.dok.dok_matrix'>
        print(adj.shape)  # (n_data, n_data)
        print(adj.nnz)  # number of edges

        save_graph_2(
            adj,
            features,
            labels,
            dataset_str=f"GenCAT_{args.dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--exp', type=str,
                        choices=[
                            'hetero_homo',
                            'attribute',
                            'scalability_edge',
                            'scalability_node_edge',
                            'classsize'],
                        default='')

    args = parser.parse_args()
    main(args)
