"""
All datasets are saved as a pytorch file(.pt format) in this repository. When you need
a dataset in anather format, you may run this file and convert the format.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append("./")


def convert_to_planetoid(dataset):
    """
    Convert data saved in .pt format to planetoid format, the same as cora, etc.
    Cora, etc. are originally in planetoid format, so there is no problem.
    target dataset: ["cornell", "texas", "wisconsin", 'blogcatalog','wiki', 'flickr', 'actor',
            'chameleon','squirrel']
    """
    from utils import convert_pt_data_into_three_component, save_graph_2
    from models.dataset_utils import DataLoader
    if dataset in ["cora", "pubmed", "citeseer"]:
        raise ValueError("conversion not allowed")

    data = DataLoader(dataset, data_dir="./data/")[0]
    adj, features, labels = convert_pt_data_into_three_component(data)
    print(type(adj))  # <class 'scipy.sparse.csr.csr_matrix'>
    print(adj.shape)  # (n_data, n_data)
    print(type(features))  # <class 'numpy.ndarray'>
    print(features.shape)  # (n_data, n_feature)
    print(type(labels))  # <class 'list'>
    print(len(labels))  # n_data
    save_graph_2(adj, features, labels, dataset_str=dataset)


def convert_to_npz(dataset):
    """
    convert to npz format
    """
    from models.dataset_utils import DataLoader
    data = DataLoader(dataset, data_dir="./data/")[0]

    x = data.x.numpy()
    edge_index = data.edge_index.numpy()
    y = data.y.numpy()
    print(edge_index.shape)

    save_npz(dataset, x, edge_index, y)


def save_npz(dataset, x, edge_index, y):
    from utils import get_path_to_top_dir

    path_top_dir = get_path_to_top_dir()
    save_dir = f"{path_top_dir}/data/{dataset}/npz/"
    os.makedirs(save_dir, exist_ok=True)
    path = f'{save_dir}{dataset}'
    np.savez(path, x=x, edge_index=edge_index, y=y)
    # npz_files = np.load(f'{path}.npz')
    # print(npz_files)
    # print(npz_files["edge_index"].shape)


def convert_to_semb(dataset):
    """
    convert to SEMB format
    https://github.com/GemsLab/StrucEmbedding-GraphLibrary/tree/master/sample-data
    """
    from models.dataset_utils import DataLoader
    data = DataLoader(dataset, data_dir="./data/")[0]

    # convert data to np.ndarray type
    # x = data.x.numpy()  # not necessary
    edge_index = data.edge_index.numpy()
    y = data.y.numpy()
    print(edge_index.shape)

    save_semb(dataset, edge_index, y)


def save_semb(dataset, edge_index, y):
    from utils import get_path_to_top_dir
    path_top_dir = get_path_to_top_dir()

    save_dir = f"{path_top_dir}/data/{dataset}/semb/"
    os.makedirs(save_dir, exist_ok=True)
    path_label = f'{save_dir}{dataset}_label.txt'
    path_edge = f'{save_dir}{dataset}.edgelist'

    df_label = pd.DataFrame({"label": y})
    df_label.to_csv(path_label, sep=" ", header=False)
    df_edge = pd.DataFrame(edge_index.T)
    df_edge.to_csv(path_edge, sep=" ", header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cornell')
    parser.add_argument('--format', default='npz', choices=["planetoid", "npz", "semb"])
    args = parser.parse_args()
    if args.format == "planetoid":
        convert_to_planetoid(args.dataset)
    elif args.format == "npz":
        convert_to_npz(args.dataset)
    elif args.format == "semb":
        convert_to_semb(args.dataset)
    else:
        raise ValueError('wrong format')
