import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear, ModuleList, ReLU, BatchNorm1d, Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import ChebConv, GMMConv, global_mean_pool, MessagePassing, APPNP
from torch_geometric.nn.models import GraphSAGE, GIN, GAT, GCN

from layers import SparseNGCNLayer, ListModule, DenseNGCNLayer, SparseLinear, MLP
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha)**np.arange(K + 1)
            TEMP[-1] = (1 - alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha)**k
        self.temp.data[-1] = (1 - self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)


class GCN_Net(GCN):
    def __init__(self, dataset, args):
        super().__init__(
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            hidden_channels=int(args.hidden),
            num_layers=args.layers,
            dropout=args.dropout,
            act=ReLU(inplace=True),
            norm=None,
            jk=args.JK
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_final = super().forward(x, edge_index)
        return F.log_softmax(x_final, dim=1)

    def set_aggr(self, aggr):
        for conv in self.convs:
            conv.aggr = aggr


class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()
        # self.conv1 = ChebConv(dataset.num_features, 32, K=2)
        # self.conv2 = ChebConv(32, dataset.num_classes, K=2)
        self.conv1 = ChebConv(dataset.num_features, int(args.hidden), K=2)
        self.conv2 = ChebConv(int(args.hidden), dataset.num_classes, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_Net(GAT):
    def __init__(self, dataset, args):
        super().__init__(
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            hidden_channels=int(args.hidden),
            num_layers=args.layers,
            dropout=args.dropout,
            act=ReLU(inplace=True),
            norm=None,
            jk=args.JK
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_final = super().forward(x, edge_index)
        return F.log_softmax(x_final, dim=1)

    def set_aggr(self, aggr):
        for conv in self.convs:
            conv.aggr = aggr


class Shadow_GAT(GAT):
    def __init__(self, dataset, args):
        super().__init__(
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            hidden_channels=int(args.hidden),
            num_layers=args.layers,
            dropout=args.dropout,
            act=ReLU(inplace=True),
            norm=None,
            jk=args.JK
        )

        del self.lin
        self.lin2 = torch.nn.Linear(2 * args.hidden, dataset.num_classes)

    def reset_parameters(self):
        super().reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch, root_n_id = data.x, data.edge_index, data.batch, data.root_n_id
        x = super().forward(x, edge_index)
        x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSAGENet(GraphSAGE):
    def __init__(self, dataset, args):
        super().__init__(
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            hidden_channels=int(args.hidden),
            num_layers=args.layers,
            dropout=args.dropout,
            act=ReLU(inplace=True),
            norm=None,
            jk=args.JK
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_final = super().forward(x, edge_index)
        return F.log_softmax(x_final, dim=1)

    def set_aggr(self, aggr):
        for conv in self.convs:
            conv.aggr = aggr


class Shadow_GraphSAGE(GraphSAGE):
    def __init__(self, dataset, args):
        super().__init__(
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            hidden_channels=int(args.hidden),
            num_layers=args.layers,
            dropout=args.dropout,
            act=ReLU(inplace=True),
            norm=None,
            jk=args.JK
        )

        del self.lin
        # for name, layer in self.named_modules():
        #     print(name, layer)

        self.lin2 = torch.nn.Linear(2 * args.hidden, dataset.num_classes)
        """
        Shadow_GraphSAGE(1433, 7, num_layers=2)
        act ReLU(inplace=True)
        convs ModuleList(
        (0): SAGEConv(1433, 64)
        (1): SAGEConv(64, 64)
        )
        convs.0 SAGEConv(1433, 64)
        convs.0.lin_l Linear(1433, 64, bias=True)
        convs.0.lin_r Linear(1433, 64, bias=False)
        convs.1 SAGEConv(64, 64)
        convs.1.lin_l Linear(64, 64, bias=True)
        convs.1.lin_r Linear(64, 64, bias=False)
        lin Linear(in_features=64, out_features=7, bias=True)
        """

    def forward(self, data):
        x, edge_index, batch, root_n_id = data.x, data.edge_index, data.batch, data.root_n_id
        x = super().forward(x, edge_index)
        x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class GINNet(GIN):
    def __init__(self, dataset, args):
        super().__init__(
            in_channels=dataset.num_features,
            out_channels=dataset.num_classes,
            hidden_channels=int(args.hidden),
            num_layers=args.layers,
            dropout=args.dropout,
            act=ReLU(inplace=True),
            norm=None,
            jk=args.JK
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_final = super().forward(x, edge_index)
        return F.log_softmax(x_final, dim=1)


class MoNet(torch.nn.Module):
    def __init__(self, dataset, args):
        # paper: Geometric deep learning on graphs and manifolds using mixture model CNNs
        # reference: https://github.com/sw-gong/MoNet/blob/master/graph/main.py
        super(MoNet, self).__init__()
        self.dropout = args.dropout
        self.conv1 = GMMConv(dataset.num_features,
                             int(args.hidden),
                             dim=2,
                             kernel_size=args.kernel_size,
                             aggr=args.aggr)
        self.conv2 = GMMConv(int(args.hidden),
                             dataset.num_classes,
                             dim=2,
                             kernel_size=args.kernel_size,
                             aggr=args.aggr)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


class SGC(torch.nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self, dataset, args):
        super(SGC, self).__init__()
        self.W = Linear(dataset.num_features, dataset.num_classes)

    def forward(self, data):
        x = self.W(data.x)
        return F.log_softmax(x, dim=1)


class MixHopNetwork(torch.nn.Module):
    """
    MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """

    def __init__(self, dataset, args):
        super(MixHopNetwork, self).__init__()
        self.args = args
        self.feature_number = dataset.num_features
        self.class_number = dataset.num_classes
        self.layers_1 = [int(args.hidden), int(args.hidden), int(args.hidden)]
        self.layers_2 = [int(args.hidden), int(args.hidden), int(args.hidden)]
        self.calculate_layer_sizes()
        self.setup_layer_structure()

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = sum(self.layers_1)
        self.abstract_feature_number_2 = sum(self.layers_2)

        self.order_1 = len(self.layers_1)
        self.order_2 = len(self.layers_2)

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional upper layers, 3 bottom layers) and dense final.
        """
        self.upper_layers = [SparseNGCNLayer(self.feature_number,
                                             self.layers_1[i - 1],
                                             i,
                                             self.args.dropout) for i in range(1,
                                                                               self.order_1 + 1)]
        self.upper_layers = ListModule(*self.upper_layers)
        self.bottom_layers = [DenseNGCNLayer(self.abstract_feature_number_1,
                                             self.layers_2[i - 1],
                                             i,
                                             self.args.dropout) for i in range(1,
                                                                               self.order_2 + 1)]
        self.bottom_layers = ListModule(*self.bottom_layers)
        self.fully_connected = torch.nn.Linear(self.abstract_feature_number_2, self.class_number)

    def calculate_group_loss(self):
        """
        Calculating the column losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            upper_column_loss = torch.norm(self.upper_layers[i].weight_matrix, dim=0)
            loss_upper = torch.sum(upper_column_loss)
            weight_loss = weight_loss + self.args.weight_decay * loss_upper
        for i in range(self.order_2):
            bottom_column_loss = torch.norm(self.bottom_layers[i].weight_matrix, dim=0)
            loss_bottom = torch.sum(bottom_column_loss)
            weight_loss = weight_loss + self.args.weight_decay * loss_bottom
        return weight_loss

    def calculate_loss(self):
        """
        Calculating the losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            loss_upper = torch.norm(self.upper_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.weight_decay * loss_upper
        for i in range(self.order_2):
            loss_bottom = torch.norm(self.bottom_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.weight_decay * loss_bottom
        return weight_loss

    def forward(self, data):
        features, normalized_adjacency_matrix = data.x_mixhop, data.adj_mixhop
        # features = feature_reader(features)
        # normalized_adjacency_matrix = create_adjacency_matrix(edge_index)

        abstract_features_1 = torch.cat([self.upper_layers[i](
            normalized_adjacency_matrix, features) for i in range(self.order_1)], dim=1)
        abstract_features_2 = torch.cat([self.bottom_layers[i](
            normalized_adjacency_matrix, abstract_features_1) for i in range(self.order_2)], dim=1)
        predictions = torch.nn.functional.log_softmax(
            self.fully_connected(abstract_features_2), dim=1)
        return predictions


class H2GCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(H2GCN, self).__init__()
        nhid = int(args.hidden)
        self.nhop = args.nhop
        self.W01 = Linear(dataset.num_features, nhid)  # ego, first layer
        self.dropout = args.dropout
        # regression
        final_layer = (self.nhop**2 + self.nhop + 1) * nhid
        self.W_final = Linear(final_layer, dataset.num_classes)

    def forward(self, data):
        # zeroth round
        x0 = [F.relu(self.W01(data.x))]
        # first round
        x1 = []
        for i in range(self.nhop):
            x1.append(torch.spmm(data.S[i], x0[0]))
        # second round
        x2 = []
        for i in range(self.nhop):
            for x_tmp in x1:
                x2.append(torch.spmm(data.S[i], x_tmp))
        x = torch.cat(x0 + x1 + x2, dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.W_final(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # return F.softmax(x, dim=1)
        return F.log_softmax(x, dim=1)


class MLPNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MLPNet, self).__init__()

        self.lin1 = Linear(dataset.num_features, int(args.hidden))
        self.lin2 = Linear(int(args.hidden), dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class FSGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(FSGNN, self).__init__()
        nlayer = args.nhop * 2 + 1
        num_features = dataset.num_features
        self.fc2 = Linear(args.hidden * nlayer, dataset.num_classes)
        self.dropout = args.dropout
        self.act_fn = torch.nn.ReLU()
        self.fc1 = ModuleList([Linear(num_features, int(args.hidden)) for _ in range(nlayer)])
        self.att = Parameter(torch.ones(nlayer))
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, data, layer_norm=True):
        mask = self.sm(self.att)
        list_out = list()
        for ind, mat in enumerate(data.SX):
            tmp_out = self.fc1[ind](mat)
            if layer_norm:
                tmp_out = F.normalize(tmp_out, p=2, dim=1)
            tmp_out = torch.mul(mask[ind], tmp_out)
            list_out.append(tmp_out)
        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.fc2(out)

        return F.log_softmax(out, dim=1)


class LINKX(torch.nn.Module):
    r"""The LINKX model from the `"Large Scale Learning on Non-Homophilous
    Graphs: New Benchmarks and Strong Simple Methods"
    <https://arxiv.org/abs/2110.14446>`_ paper

    .. math::
        \mathbf{H}_{\mathbf{A}} &= \textrm{MLP}_{\mathbf{A}}(\mathbf{A})

        \mathbf{H}_{\mathbf{X}} &= \textrm{MLP}_{\mathbf{X}}(\mathbf{X})

        \mathbf{Y} &= \textrm{MLP}_{f} \left( \sigma \left( \mathbf{W}
        [\mathbf{H}_{\mathbf{A}}, \mathbf{H}_{\mathbf{X}}] +
        \mathbf{H}_{\mathbf{A}} + \mathbf{H}_{\mathbf{X}} \right) \right)

    .. note::

        For an example of using LINKX, see `examples/linkx.py <https://
        github.com/pyg-team/pytorch_geometric/blob/master/examples/linkx.py>`_.

    Args:
        num_nodes (int): The number of nodes in the graph.
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of layers of :math:`\textrm{MLP}_{f}`.
        num_edge_layers (int): Number of layers of
            :math:`\textrm{MLP}_{\mathbf{A}}`. (default: :obj:`1`)
        num_node_layers (int): Number of layers of
            :math:`\textrm{MLP}_{\mathbf{X}}`. (default: :obj:`1`)
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
    """

    def __init__(self, dataset, args):
        super(LINKX, self).__init__()

        self.num_nodes = dataset[0].num_nodes
        self.in_channels = dataset.num_features
        self.out_channels = dataset.num_classes
        self.num_edge_layers = args.num_edge_layers

        hidden_channels = int(args.hidden)

        self.edge_lin = SparseLinear(dataset[0].num_nodes, hidden_channels)
        if self.num_edge_layers > 1:
            self.edge_norm = BatchNorm1d(hidden_channels)
            channels = [hidden_channels] * args.num_edge_layers
            self.edge_mlp = MLP(channels, dropout=0., act_first=True)

        channels = [self.in_channels] + [hidden_channels] * args.num_node_layers
        self.node_mlp = MLP(channels, dropout=0., act_first=True)

        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        channels = [hidden_channels] * args.layers + [self.out_channels]
        self.final_mlp = MLP(channels, dropout=float(args.dropout), act_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.edge_lin.reset_parameters()
        if self.num_edge_layers > 1:
            self.edge_norm.reset_parameters()
            self.edge_mlp.reset_parameters()
        self.node_mlp.reset_parameters()
        self.cat_lin1.reset_parameters()
        self.cat_lin2.reset_parameters()
        self.final_mlp.reset_parameters()

    def forward(self, data):
        """"""
        x, edge_index, edge_weight = data.x, data.edge_index, None

        out = self.edge_lin(edge_index, edge_weight)
        if self.num_edge_layers > 1:
            out = out.relu_()
            out = self.edge_norm(out)
            out = self.edge_mlp(out)

        out = out + self.cat_lin1(out)

        if x is not None:
            x = self.node_mlp(x)
            out += x
            out += self.cat_lin2(x)

        out = self.final_mlp(out.relu_())
        # return self.final_mlp(out.relu_())
        return F.log_softmax(out, dim=1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_nodes={self.num_nodes}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels})')
