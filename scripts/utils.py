import os
import random
import numpy as np

import sys
import pickle as pkl
import scipy.sparse as sp
import networkx as nx

import torch
random.seed(45)


def get_path_to_top_dir():
    path_top_dir = os.path.join(os.path.dirname(__file__), "../")
    path_top_dir = os.path.normpath(path_top_dir)
    return path_top_dir


def load_data(dataset_str):
    """
    Loads input data from dataset_dir(=f"data/{dataset_str}/{dataset_str}/raw")

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    top_dir = get_path_to_top_dir()
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    # dataset_dir = f"data/raw/{dataset_str}"
    dataset_dir = f"{top_dir}/data/{dataset_str}/{dataset_str}/raw"
    for name in names:
        with open(f"{dataset_dir}/ind.{dataset_str}.{name}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    # test_idx_reorder = parse_index_file(f"data/raw/{dataset_str}/ind.{dataset_str}.test.index")
    test_idx_reorder = parse_index_file(f"{dataset_dir}/ind.{dataset_str}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)

    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    return adj, features.toarray(), list(np.argmax(labels, axis=1))


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def save_graph(S, X, Label, dataset_str="GenCAT_test", train_val_test_ratio=[0.48, 0.32, 0.2]):
    top_dir = get_path_to_top_dir()
    # dataset_path = f"{top_dir}/data/raw_cleansed/{dataset_str}/"
    dataset_path = f"{top_dir}/data/{dataset_str}/{dataset_str}/raw/"
    os.makedirs(dataset_path, exist_ok=True)

    train_val_test_num = []
    for _ in train_val_test_ratio:
        train_val_test_num.append(int(_ * S.shape[0]))
    # print(train_val_test_num)
    train_num = train_val_test_num[0]
    val_num = train_val_test_num[1]
    test_num = train_val_test_num[2]

    ### GCN default setting ###
    # train_num = 1000
    # val_num = 500
    # test_num = 140

    val_index = list(range(S.shape[0]))[train_num:-test_num]
    test_index = list(range(S.shape[0]))[-test_num:]

    ind_list = list(range(S.shape[0]))
    random.shuffle(ind_list)

    ind_dic = dict()
    rev_dic = dict()
    for i, ind in enumerate(ind_list):
        ind_dic[i] = ind
        rev_dic[ind] = i

    X_shuffle = np.zeros(X.shape)
    for _ in range(len(X)):
        X_shuffle[ind_dic[_], :] = X[_, :]

    X_x = sp.csr_matrix(X_shuffle[:train_num])
    X_tx = sp.csr_matrix(X_shuffle[-test_num:])
    X_allx = sp.csr_matrix(X_shuffle[:-test_num])
    # print(X_x.shape, X_tx.shape, X_allx.shape)

    Label_onehot = np.identity(max(Label) + 1)[Label]
    Label_shuffle = np.zeros(Label_onehot.shape)
    for _ in range(len(Label_onehot)):
        Label_shuffle[ind_dic[_], :] = Label_onehot[_, :]

    Label_y = Label_shuffle[:train_num]
    Label_ty = Label_shuffle[-test_num:]
    Label_ally = Label_shuffle[:-test_num]

    nnz = sp.csr_matrix(S).nonzero()
    graph = dict()
    for i in range(S.shape[0]):
        graph[i] = []
    for i in range(len(nnz[0])):
        graph[ind_dic[nnz[0][i]]].append(ind_dic[nnz[1][i]])

    # Attribute
    with open(dataset_path + 'ind.' + dataset_str + '.x', 'wb') as f:
        pkl.dump(X_x, f)
    with open(dataset_path + 'ind.' + dataset_str + '.tx', 'wb') as f:
        pkl.dump(X_tx, f)
    with open(dataset_path + 'ind.' + dataset_str + '.allx', 'wb') as f:
        pkl.dump(X_allx, f)

    # Label
    with open(dataset_path + 'ind.' + dataset_str + '.y', 'wb') as f:
        pkl.dump(Label_y, f)
    with open(dataset_path + 'ind.' + dataset_str + '.ty', 'wb') as f:
        pkl.dump(Label_ty, f)
    with open(dataset_path + 'ind.' + dataset_str + '.ally', 'wb') as f:
        pkl.dump(Label_ally, f)

    # Topology
    with open(dataset_path + 'ind.' + dataset_str + '.graph', 'wb') as f:
        pkl.dump(graph, f)

    # Data split
    with open(dataset_path + 'ind.' + dataset_str + '.val.index', 'w') as f:
        for _ in val_index:
            f.write(str(_) + '\n')
    with open(dataset_path + 'ind.' + dataset_str + '.test.index', 'w') as f:
        for _ in test_index:
            f.write(str(_) + '\n')


def save_graph_2(S, X, Label, dataset_str="GenCAT_test",
                 data_dir=None, train_val_test_ratio=[0.8, 0.2]):
    """
    No split for validation data
    """
    top_dir = get_path_to_top_dir()
    # dataset_path = f"{top_dir}/data/raw_cleansed/{dataset_str}/"
    data_dir = data_dir if data_dir else f"{top_dir}/data/"
    dataset_path = f"{data_dir}/{dataset_str}/{dataset_str}/raw/"
    os.makedirs(dataset_path, exist_ok=True)

    train_val_test_num = []
    for _ in train_val_test_ratio:
        train_val_test_num.append(int(_ * S.shape[0]))
    # print(train_val_test_num)
    train_num = train_val_test_num[0]
    # val_num = train_val_test_num[1]
    test_num = train_val_test_num[1]

    ### GCN default setting ###
    # train_num = 1000
    # val_num = 500
    # test_num = 140

    # val_index = list(range(S.shape[0]))[train_num:-test_num]
    test_index = list(range(S.shape[0]))[-test_num:]

    ind_list = list(range(S.shape[0]))
    random.shuffle(ind_list)

    ind_dic = dict()
    rev_dic = dict()
    for i, ind in enumerate(ind_list):
        ind_dic[i] = ind
        rev_dic[ind] = i

    X_shuffle = np.zeros(X.shape)
    for _ in range(len(X)):
        X_shuffle[ind_dic[_], :] = X[_, :]

    X_x = sp.csr_matrix(X_shuffle[:train_num])
    X_tx = sp.csr_matrix(X_shuffle[-test_num:])
    X_allx = sp.csr_matrix(X_shuffle[:-test_num])
    # print(X_x.shape, X_tx.shape, X_allx.shape)

    Label_onehot = np.identity(max(Label) + 1)[Label]
    Label_shuffle = np.zeros(Label_onehot.shape)
    for _ in range(len(Label_onehot)):
        Label_shuffle[ind_dic[_], :] = Label_onehot[_, :]

    Label_y = Label_shuffle[:train_num]
    Label_ty = Label_shuffle[-test_num:]
    Label_ally = Label_shuffle[:-test_num]

    nnz = sp.csr_matrix(S).nonzero()
    graph = dict()
    for i in range(S.shape[0]):
        graph[i] = []
    for i in range(len(nnz[0])):
        graph[ind_dic[nnz[0][i]]].append(ind_dic[nnz[1][i]])

    # Attribute
    with open(dataset_path + 'ind.' + dataset_str + '.x', 'wb') as f:
        pkl.dump(X_x, f)
    with open(dataset_path + 'ind.' + dataset_str + '.tx', 'wb') as f:
        pkl.dump(X_tx, f)
    with open(dataset_path + 'ind.' + dataset_str + '.allx', 'wb') as f:
        pkl.dump(X_allx, f)

    # Label
    with open(dataset_path + 'ind.' + dataset_str + '.y', 'wb') as f:
        pkl.dump(Label_y, f)
    with open(dataset_path + 'ind.' + dataset_str + '.ty', 'wb') as f:
        pkl.dump(Label_ty, f)
    with open(dataset_path + 'ind.' + dataset_str + '.ally', 'wb') as f:
        pkl.dump(Label_ally, f)

    # Topology
    with open(dataset_path + 'ind.' + dataset_str + '.graph', 'wb') as f:
        pkl.dump(graph, f)

    # Data split
    # with open(dataset_path + 'ind.' + dataset_str + '.val.index', 'w') as f:
    #     for _ in val_index:
    #         f.write(str(_) + '\n')
    with open(dataset_path + 'ind.' + dataset_str + '.test.index', 'w') as f:
        for _ in test_index:
            f.write(str(_) + '\n')


def load_pt_data(dataset):
    top_dir = get_path_to_top_dir()
    if dataset in ['texas', 'cornell', 'wisconsin', 'blogcatalog', 'wiki', 'flickr', 'actor']:
        path_processed = f"{top_dir}/data/{dataset}/processed"
    elif dataset in ["chameleon", "squirrel"]:
        path_processed = f"{top_dir}/data/{dataset}/geom_gcn/processed"
    else:
        path_processed = f"{top_dir}/data/{dataset}/{dataset}/processed"
    d1 = torch.load(f"{path_processed}/data.pt")
    d2 = torch.load(f"{path_processed}/pre_filter.pt")
    d3 = torch.load(f"{path_processed}/pre_transform.pt")
    # print(type(d1))
    # print(len(d1))
    # print(d1)
    # print(type(d2))
    # print(len(d2))
    # print(d2)
    # print(type(d3))
    # print(len(d3))
    # print(d3)
    return d1, d2, d3


def convert_pt_data_into_three_component(data):
    G = nx.Graph()
    G.add_edges_from(data.edge_index.numpy().T)
    adj = nx.adjacency_matrix(G)
    features = data.x.numpy()
    labels = data.y.tolist()
    return adj, features, labels
