import os
import sys
import pickle
import shutil
import numpy as np
import os.path as osp
from itertools import repeat
from typing import Optional, Callable, List
import scipy.sparse as sp

import torch
from torch_sparse import coalesce, SparseTensor
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor
# from torch_geometric.datasets import AttributedGraphDataset  # not yet released
from torch_geometric.data import InMemoryDataset, Data, extract_zip
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
# from cSBM_dataset import dataset_ContextualSBM


class WebKB2(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        self.original_data_name = name.split("_")[-1]
        # assert self.name in [
        #     'GenCAT_cornell',
        #     'GenCAT_texas',
        #     'GenCAT_washington',
        #     'GenCAT_wisconsin']

        super(WebKB2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        # return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name}.{name}' for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        x_train = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "allx")
        x_test = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "tx")

        y_train = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "ally")
        y_test = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "ty")

        x = torch.cat((x_train, x_test), 0)
        # The number of dimensions was reduced from as many as the number of
        # classes to one dimension.
        y = torch.cat((y_train, y_test), 0).argmax(dim=1)

        graph = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "graph")

        edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))
        edge_index = to_undirected(edge_index)

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class Planetoid2(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        split (string): The type of dataset split
            (:obj:`"public"`, :obj:`"full"`, :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the
            `"Revisiting Semi-Supervised Learning with Graph Embeddings"
            <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    # url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root: str, name: str, split: str = "public",
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.split = split
        assert self.split in ['public', 'full', 'random']

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name}.{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        # for name in self.raw_file_names:
        #     download_url('{}/{}'.format(self.url, name), self.raw_dir)
        pass

    def process(self):
        data = read_planetoid_data2(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


def read_planetoid_data2(folder, prefix):
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = torch.arange(y.size(0), dtype=torch.long)
    val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    sorted_test_index = test_index.sort()[0]

    if prefix.lower() == 'citeseer':
        # There are some isolated nodes in the Citeseer graph, resulting in
        # none consecutive test indices. We need to identify them and add them
        # as zero vectors to `tx` and `ty`.
        len_test_indices = (test_index.max() - test_index.min()).item() + 1

        tx_ext = torch.zeros(len_test_indices, tx.size(1))
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = torch.zeros(len_test_indices, ty.size(1))
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    if prefix.lower() == 'nell.0.001':
        tx_ext = torch.zeros(len(graph) - allx.size(0), x.size(1))
        tx_ext[sorted_test_index - allx.size(0)] = tx

        ty_ext = torch.zeros(len(graph) - ally.size(0), y.size(1))
        ty_ext[sorted_test_index - ally.size(0)] = ty

        tx, ty = tx_ext, ty_ext

        x = torch.cat([allx, tx], dim=0)
        x[test_index] = x[sorted_test_index]

        # Creating feature vectors for relations.
        row, col, value = SparseTensor.from_dense(x).coo()
        rows, cols, values = [row], [col], [value]

        mask1 = index_to_mask(test_index, size=len(graph))
        mask2 = index_to_mask(torch.arange(allx.size(0), len(graph)),
                              size=len(graph))
        mask = ~mask1 | ~mask2
        isolated_index = mask.nonzero(as_tuple=False).view(-1)[allx.size(0):]

        rows += [isolated_index]
        cols += [torch.arange(isolated_index.size(0)) + x.size(1)]
        values += [torch.ones(isolated_index.size(0))]

        x = SparseTensor(row=torch.cat(rows), col=torch.cat(cols),
                         value=torch.cat(values))
    else:
        x = torch.cat([allx, tx], dim=0)
        x[test_index] = x[sorted_test_index]

    y = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    y[test_index] = y[sorted_test_index]

    train_mask = index_to_mask(train_index, size=y.size(0))
    val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y.size(0))

    edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


class WikipediaNetwork2(InMemoryDataset):
    r"""The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    raw_url = 'https://graphmining.ai/datasets/ptg/wiki'
    processed_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                     'geom-gcn/master')

    def __init__(self, root: str, name: str, geom_gcn_preprocess: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        # self.name = name.lower()
        self.name = name
        self.original_data_name = name.split("_")[-1]
        self.geom_gcn_preprocess = geom_gcn_preprocess
        # assert self.name in ['GenCAT_chameleon', 'GenCAT_squirrel']
        if geom_gcn_preprocess and self.name == 'crocodile':
            raise AttributeError("The dataset 'crocodile' is not available in "
                                 "case 'geom_gcn_preprocess=True'")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.original_data_name, 'geom_gcn', 'raw')
        else:
            return osp.join(self.root, self.original_data_name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.geom_gcn_preprocess:
            # return osp.join(self.root, self.name, 'geom_gcn', 'processed')
            return osp.join(self.root, self.name, 'geom_gcn', 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        if self.geom_gcn_preprocess:
            return (['out1_node_feature_label.txt', 'out1_graph_edges.txt'] +
                    [f'{self.original_data_name}_split_0.6_0.2_{i}.npz' for i in range(10)])
        else:
            return f'{self.original_data_name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        if self.geom_gcn_preprocess:
            x_train = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "allx")
            x_test = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "tx")

            y_train = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "ally")
            y_test = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "ty")

            x = torch.cat((x_train, x_test), 0)
            # The number of dimensions was reduced from as many as the number of
            # classes to one dimension.
            y = torch.cat((y_train, y_test), 0).argmax(dim=1)

            graph = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "graph")

            edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))
            edge_index = to_undirected(edge_index)

            train_masks, val_masks, test_masks = [], [], []
            for filepath in self.raw_paths[2:]:
                # get train/val/test info from orginal dataset
                f = np.load(filepath)
                train_masks += [torch.from_numpy(f['train_mask'])]
                val_masks += [torch.from_numpy(f['val_mask'])]
                test_masks += [torch.from_numpy(f['test_mask'])]
            train_mask = torch.stack(train_masks, dim=1).to(torch.bool)
            val_mask = torch.stack(val_masks, dim=1).to(torch.bool)
            test_mask = torch.stack(test_masks, dim=1).to(torch.bool)

            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        else:
            # data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
            # x = torch.from_numpy(data['features']).to(torch.float)
            # edge_index = torch.from_numpy(data['edges']).to(torch.long)
            # edge_index = edge_index.t().contiguous()
            # edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
            # y = torch.from_numpy(data['target']).to(torch.float)

            # data = Data(x=x, edge_index=edge_index, y=y)
            pass

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


class AttributedGraphDataset(InMemoryDataset):
    r"""A variety of attributed graph datasets from the
    `"Scaling Attributed Network Embedding to Massive Graphs"
    <https://arxiv.org/abs/2009.00826>`_ paper.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Wiki"`, :obj:`"Cora"`
            :obj:`"CiteSeer"`, :obj:`"PubMed"`, :obj:`"BlogCatalog"`,
            :obj:`"PPI"`, :obj:`"Flickr"`, :obj:`"Facebook"`, :obj:`"Twitter"`,
            :obj:`"TWeibo"`, :obj:`"MAG"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    datasets = {
        'wiki': '1EPhlbziZTQv19OsTrKrAJwsElbVPEbiV',
        'cora': '1FyVnpdsTT-lhkVPotUW8OVeuCi1vi3Ey',
        'citeseer': '1d3uQIpHiemWJPgLgTafi70RFYye7hoCp',
        'pubmed': '1DOK3FfslyJoGXUSCSrK5lzdyLfIwOz6k',
        'blogcatalog': '178PqGqh67RUYMMP6-SoRHDoIBh8ku5FS',
        'ppi': '1dvwRpPT4gGtOcNP_Q-G1TKl9NezYhtez',
        'flickr': '1tZp3EB20fAC27SYWwa-x66_8uGsuU62X',
        'facebook': '12aJWAGCM4IvdGI2fiydDNyWzViEOLZH8',
        'twitter': '1fUYggzZlDrt9JsLsSdRUHiEzQRW1kSA4',
        'tweibo': '1-2xHDPFCsuBuFdQN_7GLleWa8R_t50qU',
        'mag': '1ggraUMrQgdUyA3DjSRzzqMv0jFkU65V5',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['attrs.npz', 'edgelist.txt', 'labels.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        from google_drive_downloader import GoogleDriveDownloader as gdd
        path = osp.join(self.raw_dir, f'{self.name}.zip')
        gdd.download_file_from_google_drive(self.datasets[self.name], path)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        path = osp.join(self.raw_dir, f'{self.name}.attr')
        if self.name == 'mag':
            path = osp.join(self.raw_dir, self.name)
        for name in self.raw_file_names:
            os.rename(osp.join(path, name), osp.join(self.raw_dir, name))
        shutil.rmtree(path)

    def process(self):
        import pandas as pd

        x = sp.load_npz(self.raw_paths[0])
        # if x.shape[-1] > 10000 or self.name == 'mag':
        #     x = SparseTensor.from_scipy(x).to(torch.float)
        # else:
        #     x = torch.from_numpy(x.todense()).to(torch.float)
        x = torch.from_numpy(x.todense()).to(torch.float)

        df = pd.read_csv(self.raw_paths[1], header=None, sep=None,
                         engine='python')
        edge_index = torch.from_numpy(df.values).t().contiguous()

        with open(self.raw_paths[2], 'r') as f:
            ys = f.read().split('\n')[:-1]
            ys = [[int(y) - 1 for y in row.split()[1:]] for row in ys]
            multilabel = max([len(y) for y in ys]) > 1

        if not multilabel:
            y = torch.tensor(ys).view(-1)
        else:
            num_classes = max([y for row in ys for y in row]) + 1
            y = torch.zeros((len(ys), num_classes), dtype=torch.float)
            for i, row in enumerate(ys):
                for j in row:
                    y[i, j] = 1.

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'


class AttributedGraphDataset2(InMemoryDataset):
    r"""A variety of attributed graph datasets from the
    `"Scaling Attributed Network Embedding to Massive Graphs"
    <https://arxiv.org/abs/2009.00826>`_ paper.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Wiki"`, :obj:`"Cora"`
            :obj:`"CiteSeer"`, :obj:`"PubMed"`, :obj:`"BlogCatalog"`,
            :obj:`"PPI"`, :obj:`"Flickr"`, :obj:`"Facebook"`, :obj:`"Twitter"`,
            :obj:`"TWeibo"`, :obj:`"MAG"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    datasets = {
        'GenCAT_wiki': '1EPhlbziZTQv19OsTrKrAJwsElbVPEbiV',
        'cora': '1FyVnpdsTT-lhkVPotUW8OVeuCi1vi3Ey',
        'citeseer': '1d3uQIpHiemWJPgLgTafi70RFYye7hoCp',
        'pubmed': '1DOK3FfslyJoGXUSCSrK5lzdyLfIwOz6k',
        'GenCAT_blogcatalog': '178PqGqh67RUYMMP6-SoRHDoIBh8ku5FS',
        'ppi': '1dvwRpPT4gGtOcNP_Q-G1TKl9NezYhtez',
        'GenCAT_flickr': '1tZp3EB20fAC27SYWwa-x66_8uGsuU62X',
        'facebook': '12aJWAGCM4IvdGI2fiydDNyWzViEOLZH8',
        'twitter': '1fUYggzZlDrt9JsLsSdRUHiEzQRW1kSA4',
        'tweibo': '1-2xHDPFCsuBuFdQN_7GLleWa8R_t50qU',
        'mag': '1ggraUMrQgdUyA3DjSRzzqMv0jFkU65V5',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        self.original_name = name.split("_")[-1]
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['attrs.npz', 'edgelist.txt', 'labels.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        x_train = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "allx")
        x_test = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "tx")

        y_train = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "ally")
        y_test = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "ty")

        x = torch.cat((x_train, x_test), 0)
        # The number of dimensions was reduced from as many as the number of
        # classes to one dimension.
        y = torch.cat((y_train, y_test), 0).argmax(dim=1)

        graph = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "graph")

        edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))
        edge_index = to_undirected(edge_index)

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'


class Actor2(InMemoryDataset):
    r"""The actor-only induced subgraph of the film-director-actor-writer
    network used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Each node corresponds to an actor, and the edge between two nodes denotes
    co-occurrence on the same Wikipedia page.
    Node features correspond to some keywords in the Wikipedia pages.
    The task is to classify the nodes into five categories in term of words of
    actor's Wikipedia.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = "GenCAT_actor"
        self.original_data_name = "actor"
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.original_data_name, 'raw')

    @property
    def raw_file_names(self) -> List[str]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
                ] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):

        with open(self.raw_paths[0], 'r') as f:
            x_train = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "allx")
            x_test = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "tx")

            y_train = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "ally")
            y_test = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "ty")

            x = torch.cat((x_train, x_test), 0)
            # The number of dimensions was reduced from as many as the number of
            # classes to one dimension.
            y = torch.cat((y_train, y_test), 0).argmax(dim=1)

            graph = read_file(f"{self.root}/{self.name}/{self.name}/raw/", self.name, "graph")

            edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))
            edge_index = to_undirected(edge_index)

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


def read_file(folder, prefix, name):
    """
    copied from the following repository
    https://github.com/pyg-team/pytorch_geometric/blob/614e3a6402a542bfdd0b9ecf4ec37607b4b61756/torch_geometric/io/planetoid.py#L17
    """
    path = osp.join(folder, 'ind.{}.{}'.format(prefix, name))

    if name == 'test.index':
        return read_txt_array(path, dtype=torch.long)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = torch.Tensor(out)
    return out


def edge_index_from_dict(graph_dict, num_nodes=None, do_coalesce=True):
    """
    copied from the pytorch_geometric repository
    """
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    print("# graph_dict size", len(graph_dict.items()))
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    if do_coalesce:
        # NOTE: There are some duplicated edges and self loops in the datasets.
        #       Other implementations do not remove them!
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    print(edge_index.shape)
    return edge_index


def index_to_mask(index, size):
    """
    copied from the pytorch_geometric repository
    """
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def DataLoader(name, data_dir="../data/"):
    if 'cSBM_data' in name:
        # dataset = dataset_ContextualSBM(data_dir, name=name)
        pass
    else:
        if "GenCAT" in name:
            pass
        else:
            name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(data_dir + name, name, transform=T.NormalizeFeatures())

    elif ("cora" in name) | ("pubmed" in name) | ("citeseer" in name):
        dataset = Planetoid2(data_dir + name, name, transform=T.NormalizeFeatures())

    elif name in ['chameleon', 'squirrel']:
        # dataset = dataset_heterophily(
        #     root='../data/', name=name, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root=data_dir, name=name, transform=T.NormalizeFeatures())
    elif ("chameleon" in name) | ("squirrel" in name):
        dataset = WikipediaNetwork2(
            root=data_dir, name=name, transform=T.NormalizeFeatures())

    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root=data_dir,
                        name=name, transform=T.NormalizeFeatures())
    elif ("texas" in name) | ("cornell" in name) | ("wisconsin" in name):
        dataset = WebKB2(root=data_dir,
                         name=name, transform=T.NormalizeFeatures())
    elif name in ['blogcatalog', 'wiki', 'flickr']:
        dataset = AttributedGraphDataset(
            root=data_dir, name=name, transform=T.NormalizeFeatures())
    elif name in ['GenCAT_blogcatalog', 'GenCAT_wiki', 'GenCAT_flickr']:
        dataset = AttributedGraphDataset2(
            root=data_dir, name=name, transform=T.NormalizeFeatures())
    # elif name in ['flickr']:
    #     dataset = Flickr(root=f'../data/{name}/', transform=T.NormalizeFeatures())
    elif name in ['actor']:
        dataset = Actor(root=f'{data_dir}{name}/', transform=T.NormalizeFeatures())
    elif name in ['GenCAT_actor']:
        dataset = Actor2(root=data_dir, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset
