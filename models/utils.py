import torch
import random
from time import perf_counter
import numpy as np
from scipy import sparse
from torch_sparse import SparseTensor, remove_diag, set_diag, spmm
from torch_sparse import sum as ts_sum
from torch_sparse import mul as ts_mul

from torch_geometric.utils import degree

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(1)
# DEVICE = torch.device('cpu')
SEED = 42


def dense2sparseTensor(adj):
    adj_coo = adj.to('cpu').to_sparse()
    adj = torch.sparse.FloatTensor(
        torch.stack([adj_coo.indices()[0], adj_coo.indices()[1]]).to(DEVICE),
        adj_coo.values().to(DEVICE),
        adj.shape)
    return adj


def fix_seed(seed=SEED):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn + val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn + val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data


def hgcn_precompute(n, edge_index, nhop=2):
    # DEVICE = torch.device('cpu')
    t = perf_counter()
    # convert edge_index to sparse matrix of torch
    # adj = torch.sparse_coo_tensor(edge_index, torch.ones(
    # edge_index.shape[1]), (n, n)).float().to(DEVICE)
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(
        edge_index.shape[1]), (n, n)).float()
    # remove self-loop
    adj = remove_selfloop(adj).to(DEVICE)
    adj = normalization(adj)
    # adj = dense2sparseTensor(adj)
    S = [dense2sparseTensor(adj)]
    for i in range(1, nhop):
        # 2-hop neighbor
        S.append(torch.spmm(S[i - 1], adj))  # compute exact (i+1)-hop neighbor
        S[i][S[i] > 0] = 1
        S[i] = S[i] - torch.eye(S[i].size(0)).to(DEVICE)  # remove self-loop
        # S[i] = S[i] - torch.eye(S[i].size(0))
        for j in range(i - 1):
            S[i] = S[i] - S[j]  # remove connection within nhop-1 to get exact nhop neighbor
        S[i][S[i] < 0] = 0
        # normalization
        S[i] = normalization(S[i].to_sparse().to(DEVICE))
        # S[i] = normalization(S[i].to_sparse())
    precompute_time = perf_counter() - t
    return S, precompute_time


def normalization(adj):
    degrees = torch.sparse.sum(adj, dim=0).to_dense()
    # adj = adj.to(DEVICE)
    D = torch.pow(degrees, -0.5).to(DEVICE)
    # D = torch.pow(degrees, -0.5)
    D = torch.nan_to_num(D, nan=0.0, posinf=0.0)
    D = torch.diag(D)
    adj_out = torch.spmm(adj, D)
    D = D.to('cpu').to_sparse().to(DEVICE)
    adj_out = torch.spmm(D, adj_out)
    torch.cuda.empty_cache()
    return adj_out


def minibatch_normalization(adj, N, k=1000000):
    deg = ts_sum(adj, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    deg_inv_col = deg_inv_sqrt.view(N, 1).to(DEVICE)
    deg_inv_row = deg_inv_sqrt.view(1, N).to(DEVICE)

    adj = adj.coo()
    for i in range(len(adj[0]) // k + 1):
        tmp = SparseTensor(row=adj[0][i * k:(i + 1) * k],
                           col=adj[1][i * k:(i + 1) * k],
                           # value=adj[2][i*k:(i+1)*k],
                           sparse_sizes=(N, N)).to(DEVICE)
        tmp = ts_mul(tmp, deg_inv_col)
        tmp = ts_mul(tmp, deg_inv_row).to('cpu').coo()
        if i == 0:
            adj_t = [tmp[0], tmp[1], tmp[2]]
        else:
            for _ in range(3):
                adj_t[_] = torch.concat([adj_t[_], tmp[_]], dim=0)
    adj_t = SparseTensor(row=adj_t[0],
                         col=adj_t[1],
                         value=adj_t[2],
                         sparse_sizes=(N, N))
    del deg_inv_col, deg_inv_row, tmp
    torch.cuda.empty_cache()
    return adj_t


def add_selfloop(adj):
    adj = adj.to_dense()
    for i in range(adj.shape[0]):
        if adj[i, i] != 1:
            adj[i, i] = 1
    return adj.to_sparse()


def remove_selfloop(adj):
    adj = adj.to_dense()
    for i in range(adj.shape[0]):
        if adj[i, i] != 0:
            adj[i, i] = 0
    return adj.to_sparse()


def sgc_precompute(features, edge_index, degree=2):
    t = perf_counter()
    n = features.shape[0]
    # convert edge_index to sparse matrix of torch
    # adj = torch.sparse_coo_tensor(edge_index, torch.ones(
    # edge_index.shape[1]), (n, n)).float()
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(n, n))
    # add self-loop
    adj = set_diag(adj, 1)
    adj = minibatch_normalization(adj, n).coo()
    adj = torch.sparse.FloatTensor(torch.stack([adj[0], adj[1]]), adj[2], [n, n]).to(DEVICE)
    # adj = add_selfloop(adj).to(DEVICE)
    # adj = normalization(adj)
    # adj = dense2sparseTensor(adj)
    for i in range(degree):
        features = torch.spmm(adj, features.to(DEVICE))
    precompute_time = perf_counter() - t
    return features, precompute_time


def monet_precompute(data):
    t = perf_counter()
    # reference:
    # https://github.com/sw-gong/MoNet/blob/master/graph/main.py
    # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/nets/SBMs_node_classification/mo_net.py
    row, col = data.edge_index
    deg = degree(col, data.num_nodes).to(DEVICE)
    # print(deg.shape)  # torch.Size([num_nodes])
    # print(deg.sum())  # equal to num_edges
    # to avoid zero division in case deg is 0, we add constant '1' in
    # all node degrees denoting self-loop
    edge_attr = torch.stack(
        [1 / torch.sqrt(deg[row] + 1), 1 / torch.sqrt(deg[col] + 1)], dim=-1)
    # print(edge_attr.shape)  # torch.Size([num_edges, 2])
    precompute_time = perf_counter() - t
    return edge_attr, precompute_time


def fsgnn_precompute(features, edge_index, nhop):
    t = perf_counter()
    n = features.shape[0]
    # convert edge_index to sparse matrix of torch

    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(n, n))
    # add self-loop
    adj_ = set_diag(adj, 1)
    adj_ = minibatch_normalization(adj_, n).coo()
    adj_ = torch.sparse.FloatTensor(torch.stack([adj_[0], adj_[1]]), adj_[2], [n, n]).to(DEVICE)

    # adj = torch.sparse_coo_tensor(edge_index, torch.ones(
    #     edge_index.shape[1]), (n, n)).float()
    # remove self-loop
    adj_i = remove_diag(adj, 0)
    adj_i = minibatch_normalization(adj_i, n).coo()
    adj_i = torch.sparse.FloatTensor(torch.stack([adj_i[0], adj_i[1]]), adj_i[2], [n, n]).to(DEVICE)
    torch.cuda.empty_cache()

    features = features.to(DEVICE)
    features = preprocess_features(features)
    SX = [features]
    tmp = features
    tmp_i = features
    for _ in range(nhop):
        tmp = torch.spmm(adj_, tmp)
        SX.append(tmp)
        tmp_i = torch.spmm(adj_i, tmp_i)
        SX.append(tmp_i)
    precompute_time = perf_counter() - t
    return features, SX, precompute_time


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1).to('cpu'))
    rowsum = (rowsum == 0) * 1 + rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    n = len(r_inv)
    r_mat_inv = torch.sparse.FloatTensor(torch.stack([torch.tensor(np.arange(n)), torch.tensor(
        np.arange(n))]), torch.FloatTensor(r_inv), [n, n]).to(DEVICE)
    # r_mat_inv = torch.diag().to(DEVICE)
    features = torch.spmm(r_mat_inv, features)
    return features


def create_adjacency_matrix(edge_index):
    """
    Creating a sparse adjacency matrix.
    :param graph: NetworkX object.
    :return A: Adjacency matrix.
    """
    # index_1 = [edge[0] for edge in graph.edges()] + [edge[1] for edge in graph.edges()]
    # index_2 = [edge[1] for edge in graph.edges()] + [edge[0] for edge in graph.edges()]
    index_1 = list(edge_index[0]) + list(edge_index[1])
    index_2 = list(edge_index[1]) + list(edge_index[0])

    values = [1 for index in index_1]
    node_count = max(max(index_1) + 1, max(index_2) + 1)
    A = sparse.coo_matrix((values, (index_1, index_2)),
                          shape=(node_count, node_count),
                          dtype=np.float32)

    adjacency_matrix = dict()
    ind = np.concatenate([A.row.reshape(-1, 1), A.col.reshape(-1, 1)], axis=1)
    adjacency_matrix["indices"] = torch.LongTensor(ind.T)
    adjacency_matrix["values"] = torch.FloatTensor(A.data)
    return adjacency_matrix


def feature_reader(features):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param path: Path to the JSON file.
    :return out_features: Dict with index and value tensor.
    """
    # features = json.load(open(path))
    # index_1 = [int(k) for k, v in features.items() for fet in v]
    # index_2 = [int(fet) for k, v in features.items() for fet in v]
    # index_1 = [(range(len(features.shape[0] * features.shape[1])))]
    # index_2 = list(features)
    # values = [1.0] * len(index_1)
    # nodes = [int(k) for k, v in features.items()]
    # node_count = max(nodes) + 1
    # feature_count = max(index_2) + 1
    # features = sparse.coo_matrix((values, (index_1, index_2)),
    #                              shape=(node_count, feature_count),
    #                              dtype=np.float32)
    # print("##", features.shape)
    # print("##", (features > 0).sum())
    dimensions = features.shape

    features = sparse.coo_matrix(features.cpu())
    # print("##", features.data.shape)
    # print("##", features.data.sum())

    out_features = dict()
    ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
    out_features["indices"] = torch.LongTensor(ind.T)
    out_features["values"] = torch.FloatTensor(features.data)
    out_features["dimensions"] = dimensions
    return out_features


def mixhop_precompute(features, edge_index):
    t = perf_counter()
    out_features = feature_reader(features)
    adjacency_matrix = create_adjacency_matrix(edge_index)

    precompute_time = perf_counter() - t
    return out_features, adjacency_matrix, precompute_time
