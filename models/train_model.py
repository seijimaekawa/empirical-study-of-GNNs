import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from time import perf_counter

from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.loader import GraphSAINTRandomWalkSampler, ShaDowKHopSampler

from dataset_utils import DataLoader
from utils import random_planetoid_splits, hgcn_precompute, fix_seed, sgc_precompute, monet_precompute, fsgnn_precompute, mixhop_precompute
from GNN_models import GCN_Net, GAT_Net, APPNP_Net, ChebNet, GPRGNN, GraphSAGENet, GINNet, SGC, MixHopNetwork, H2GCN, MLPNet, MoNet, FSGNN, Shadow_GAT, Shadow_GraphSAGE, LINKX

os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fix_seed()


net_dict = {"GCN": GCN_Net,
            "GAT": GAT_Net,
            "APPNP": APPNP_Net,
            "ChebNet": ChebNet,
            "GPRGNN": GPRGNN,
            "GraphSAGE": GraphSAGENet,
            "GIN": GINNet,
            "SGC": SGC,
            "MixHop": MixHopNetwork,
            "H2GCN": H2GCN,
            "MLP": MLPNet,
            "MoNet": MoNet,
            "FSGNN": FSGNN,
            "GraphSAINT-GAT": GAT_Net,
            "GraphSAINT-GraphSAGE": GraphSAGENet,
            "JK-GCN": GCN_Net,
            "JK-GAT": GAT_Net,
            "JK-GraphSAGE": GraphSAGENet,
            "Shadow-GAT": Shadow_GAT,
            "Shadow-GraphSAGE": Shadow_GraphSAGE,
            "LINKX": LINKX
            }


def RunExp(args, dataset, data, Net, percls_trn, val_lb):
    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    if "GraphSAINT" in args.net:
        row, col = data.edge_index
        data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
        loader = GraphSAINTRandomWalkSampler(
            data.to("cpu"), batch_size=args.root, walk_length=args.walk_length, sample_coverage=args.sample_coverage)

    elif "Shadow" in args.net:
        row, col = data.edge_index
        data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
        kwargs = {'batch_size': args.batch_size, "num_workers": 0}
        train_loader_for_training = ShaDowKHopSampler(data, depth=args.depth, num_neighbors=args.num_neighbors,
                                                      node_idx=data.train_mask, **kwargs)

        # kwargs1 = {'batch_size': int(data.x.shape[0]), "num_workers": 0}
        kwargs1 = {'batch_size': args.batch_size, "num_workers": 0}

        train_loader = ShaDowKHopSampler(data, depth=args.depth, num_neighbors=args.num_neighbors,
                                         node_idx=data.train_mask, **kwargs1)
        val_loader = ShaDowKHopSampler(data, depth=args.depth, num_neighbors=args.num_neighbors,
                                       node_idx=data.val_mask, **kwargs1)
        test_loader = ShaDowKHopSampler(data, depth=args.depth, num_neighbors=args.num_neighbors,
                                        node_idx=data.test_mask, **kwargs1)

    def train(model, optimizer, data, dprate):
        if "GraphSAINT" in args.net:
            use_normalization = False
            model.train()
            model.set_aggr('add' if use_normalization else 'mean')

            total_loss = total_examples = 0
            for data_ in loader:
                # print(data_.x.shape)
                data_ = data_.to(DEVICE)
                optimizer.zero_grad()

                if use_normalization:
                    edge_weight = data_.edge_norm * data_.edge_weight
                    out = model(data_, edge_weight)
                    loss = F.nll_loss(out, data_.y, reduction='none')
                    loss = (loss * data_.node_norm)[data_.train_mask].sum()
                else:
                    out = model(data_)
                    loss = F.nll_loss(out[data_.train_mask], data_.y[data_.train_mask])

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data_.num_nodes
                total_examples += data_.num_nodes
                loss = total_loss / total_examples
        elif "Shadow" in args.net:
            model.train()
            total_loss = total_examples = 0
            for mini_data in train_loader_for_training:
                mini_data = mini_data.to(DEVICE)
                optimizer.zero_grad()
                out = model(mini_data)
                # loss = F.cross_entropy(out, mini_data.y)
                nll = F.nll_loss(out, mini_data.y)
                loss = nll
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * mini_data.num_graphs
                total_examples += mini_data.num_graphs
            _ = total_loss / total_examples
        else:
            model.train()
            optimizer.zero_grad()
            if args.net == "MixHop":
                base_run = False
                out = model(data)[data.train_mask]
                loss = F.nll_loss(out, data.y[data.train_mask])
                if base_run:
                    loss = loss + model.calculate_group_loss()
                else:
                    loss = loss + model.calculate_loss()
                # print("loss:", loss)
            else:
                out = model(data)[data.train_mask]
                loss = F.nll_loss(out, data.y[data.train_mask])
                # loss = nll

            loss.backward()

            optimizer.step()

    def test(model, data):
        if "Shadow" in args.net:
            return test_shadow(model)
        else:
            model.eval()
            if "GraphSAINT" in args.net:
                model.set_aggr('mean')
            logits = model(data)

            accs, losses, preds, f1_micros, f1_macros = [], [], [], [], []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = logits[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

                f1_micro = f1_score(data.y[mask].to("cpu"), pred.to("cpu"), average='micro')
                f1_macro = f1_score(data.y[mask].to("cpu"), pred.to("cpu"), average='macro')

                loss = F.nll_loss(logits[mask], data.y[mask])

                preds.append(pred.detach().cpu())
                accs.append(acc)
                losses.append(loss.detach().cpu())
                f1_micros.append(f1_micro)
                f1_macros.append(f1_macro)
            return accs, preds, losses, f1_micros, f1_macros

    def test_shadow(model):
        # Shadow GNN
        model.eval()
        # total_correct = total_examples = 0
        accs, losses, preds, f1_micros, f1_macros = [], [], [], [], []
        for shadow_loader in [train_loader, val_loader, test_loader]:

            total_correct = total_loss = total_examples = 0
            pred_list = y_list = []
            for mini_data in shadow_loader:
                mini_data = mini_data.to(DEVICE)
                logits = model(mini_data)
                loss = F.nll_loss(logits, mini_data.y)
                total_loss += float(loss) * mini_data.num_graphs
                pred = logits.max(1)[1]
                pred_list = pred_list + pred.to("cpu").tolist()
                y_list = y_list + mini_data.y.to("cpu").tolist()

                n_correct = pred.eq(mini_data.y).sum().item()  # 正解数
                total_correct += n_correct
                total_examples += mini_data.num_graphs
            acc = total_correct / total_examples
            final_loss = total_loss / total_examples
            accs.append(acc)
            losses.append(final_loss)
            preds.append(pred_list)
            f1_micro = f1_score(y_list, pred_list, average='micro')
            f1_macro = f1_score(y_list, pred_list, average='macro')
            f1_micros.append(f1_micro)
            f1_macros.append(f1_macro)
        return accs, preds, losses, f1_micros, f1_macros

    gnn = Net(dataset, args)

    if "Shadow" in args.net:
        model = gnn.to(DEVICE)
    else:
        model, data = gnn.to(DEVICE), data.to(DEVICE)
    print("features(x):", data.x.shape)
    print("edge_index:", data.edge_index.shape)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    elif args.net in ['FSGNN']:
        optimizer = torch.optim.Adam([
            {'params': model.fc2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.fc1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.att, 'weight_decay': args.att_weight_decay, 'lr': args.lr},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    t = perf_counter()  # Start measuring time

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [tmp_train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss], [
            tmp_train_f1_micro, tmp_val_f1_micro, tmp_test_f1_micro], [
                tmp_train_f1_macro, tmp_val_f1_macro, tmp_test_f1_macro] = test(model, data)

        if val_loss < best_val_loss:
            train_acc = tmp_train_acc
            train_f1_micro = tmp_train_f1_micro
            train_f1_macro = tmp_train_f1_macro

            best_val_acc = val_acc
            best_val_loss = val_loss
            val_f1_micro = tmp_val_f1_micro
            val_f1_macro = tmp_val_f1_macro

            test_acc = tmp_test_acc
            test_f1_micro = tmp_test_f1_micro
            test_f1_macro = tmp_test_f1_macro

            if args.net == 'GPRGNN':
                TEST = gnn.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
    stop_epoch = epoch

    total_time = perf_counter() - t

    return test_acc, best_val_acc, train_acc, test_f1_micro, val_f1_micro, train_f1_micro, test_f1_macro, val_f1_macro, train_f1_macro, stop_epoch, Gamma_0, total_time


def get_data_dir(args):
    if args.exp:
        data_dir = f"../data/GenCAT_Exp_{args.exp}/"
    else:
        data_dir = "../data/"

    return data_dir


def main(args, experiment=None):
    gnn_name = args.net
    Net = net_dict[gnn_name]
    data_dir = get_data_dir(args)

    dataset = DataLoader(args.dataset, data_dir)
    data = dataset[0]

    # add dummy GPU access
    dummy_tensor = torch.tensor([0]).to(DEVICE)

    if gnn_name == "H2GCN":
        data.S, precompute_time = hgcn_precompute(data.x.shape[0], data.edge_index, nhop=args.nhop)

    elif gnn_name == "MoNet":
        data.edge_attr, precompute_time = monet_precompute(data)

    elif gnn_name == "SGC":
        data.x, precompute_time = sgc_precompute(data.x, data.edge_index)
    elif gnn_name == "FSGNN":
        data.x, data.SX, precompute_time = fsgnn_precompute(data.x, data.edge_index, nhop=args.nhop)
    elif gnn_name == "MixHop":
        data.x_mixhop, data.adj_mixhop, precompute_time = mixhop_precompute(data.x, data.edge_index)

    else:
        precompute_time = 0
    print("precompute_time:", precompute_time)

    RPMAX = args.RPMAX
    Init = args.Init

    Gamma_0 = None
    train_rate = args.train_rate
    val_rate = args.val_rate
    test_rate = 1 - (train_rate + val_rate)
    percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(val_rate * len(data.y)))
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
    print('True Label rate: ', TrueLBrate)

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    Results0 = []

    metrics = {}
    for RP in tqdm(range(RPMAX)):
        test_acc, best_val_acc, train_acc, test_f1_micro, val_f1_micro, train_f1_micro, test_f1_macro, val_f1_macro, train_f1_macro, num_epoch, Gamma_0, total_time = RunExp(
            args, dataset, data, Net, percls_trn, val_lb)
        total_time = total_time + precompute_time
        epoch_time = total_time / num_epoch if num_epoch > 0 else total_time
        Results0.append([test_acc,
                         best_val_acc,
                         train_acc,
                         test_f1_micro,
                         val_f1_micro,
                         train_f1_micro,
                         test_f1_macro,
                         val_f1_macro,
                         train_f1_macro,
                         total_time / 100,
                         epoch_time / 100,
                         num_epoch / 100
                         ])
        metrics["iteration"] = RP
        metrics["test_accuracy"] = test_acc
        metrics["validation_accuracy"] = best_val_acc
        metrics["train_accuracy"] = train_acc
        metrics["total_time"] = total_time
        metrics["epoch_time"] = epoch_time
        metrics["num_epoch"] = num_epoch
        print(metrics)

        print("test_f1_macro", test_f1_macro)
        print("test_f1_micro", test_f1_micro)
        if experiment:
            experiment.log_metrics({"test_acc": test_acc * 100,
                                    "test_f1_micro": test_f1_micro * 100,
                                    "test_f1_macro": test_f1_macro * 100})

    test_acc_mean, val_acc_mean, train_acc_mean, test_f1_micro_mean, val_f1_micro_mean, train_f1_micro_mean, test_f1_macro_mean, val_f1_macro_mean, train_f1_macro_mean, total_time_mean, epoch_time_mean, epoch_mean = np.mean(
        Results0, axis=0) * 100
    # test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
    test_acc_std, val_acc_std, train_acc_std, test_f1_micro_std, val_f1_micro_std, train_f1_micro_std, test_f1_macro_std, val_f1_macro_std, train_f1_macro_std, total_time_std, epoch_time_std, epoch_std = np.sqrt(
        np.var(Results0, axis=0)) * 100

    print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
    print(
        f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')

    if experiment:
        experiment.log_metrics({"test_acc_mean": test_acc_mean,
                                "test_f1_micro_mean": test_f1_micro_mean,
                                "test_f1_macro_mean": test_f1_macro_mean})
        args = vars(args)
        args["test_acc_mean"] = test_acc_mean
        args["test_acc_std"] = test_acc_std
        args["test_f1_micro_mean"] = test_f1_micro_mean
        args["test_f1_micro_std"] = test_f1_micro_std
        args["test_f1_macro_mean"] = test_f1_macro_mean
        args["test_f1_macro_std"] = test_f1_macro_std

        args["val_acc_mean"] = val_acc_mean
        args["val_acc_std"] = val_acc_std
        args["val_f1_micro_mean"] = val_f1_micro_mean
        args["val_f1_micro_std"] = val_f1_micro_std
        args["val_f1_macro_mean"] = val_f1_macro_mean
        args["val_f1_macro_std"] = val_f1_macro_std

        args["train_acc_mean"] = train_acc_mean
        args["train_acc_std"] = train_acc_std
        args["train_f1_micro_mean"] = train_f1_micro_mean
        args["train_f1_micro_std"] = train_f1_micro_std
        args["train_f1_macro_mean"] = train_f1_macro_mean
        args["train_f1_macro_std"] = train_f1_macro_std

        args["total_time_mean"] = total_time_mean
        args["total_time_std"] = total_time_std
        args["epoch_time_mean"] = epoch_time_mean
        args["epoch_time_std"] = epoch_time_std
        args["epoch_mean"] = epoch_mean
        args["epoch_std"] = epoch_std

        args["test_rate"] = test_rate
        experiment.log_parameters(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.025)
    parser.add_argument('--val_rate', type=float, default=0.025)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str, default='PPR',
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'])
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', type=str, default='GCN',
                        choices=[
                            'GCN',
                            'GAT',
                            'APPNP',
                            'ChebNet',
                            'GPRGNN',
                            'GraphSAGE',
                            'GIN',
                            'SGC',
                            'MixHop',
                            'H2GCN',
                            'MLP',
                            'MoNet',
                            'FSGNN',
                            "JK-GCN",
                            "JK-GAT",
                            "JK-GraphSAGE",
                            "GraphSAINT-GAT",
                            "GraphSAINT-GraphSAGE",
                            "Shadow-GAT",
                            "Shadow-GraphSAGE",
                            "LINKX"],
                        )
    parser.add_argument('--JK', type=str, default='last',
                        choices=['last', 'cat', 'max', 'lstm'])
    parser.add_argument('--layers', type=int, default=2)

    parser.add_argument('--nhop', type=int, default=2)  # H2GCN

    parser.add_argument('--kernel_size', type=int, default=16)  # MoNet
    parser.add_argument('--aggr', type=str, default="mean", choices=["mean", "add", "max"])  # MoNet

    parser.add_argument('--parameter_search', action='store_true')  # parameter experiment
    parser.add_argument('--cometml', action='store_true')  # log with cometml

    parser.add_argument('--exp', type=str, default='',
                        choices=[
                            'hetero_homo',
                            'attribute',
                            'scalability_edge',
                            'scalability_node_edge',
                            'classsize']
                        )
    parser.add_argument('--walk_length', type=int, default=2)  # graphSAINT
    parser.add_argument('--sample_coverage', type=int, default=50)  # graphSAINT
    parser.add_argument('--root', type=int, default=2000)  # graphSAINT

    parser.add_argument('--batch_size', type=int, default=512)  # ShadowGNN
    parser.add_argument('--depth', type=int, default=2)  # ShadowGNN
    parser.add_argument('--num_neighbors', type=int, default=10)  # ShadowGNN

    # use whether conditional best parameter or not
    parser.add_argument('--conditional', action='store_true')

    parser.add_argument('--att_weight_decay', type=float, default=0.01)  # FSGNN

    parser.add_argument('--full_search_params', action='store_true')

    parser.add_argument('--num_edge_layers', type=int, default=2)   # LINKX
    parser.add_argument('--num_node_layers', type=int, default=2)   # LINKX

    # supplementary experiment
    parser.add_argument('--supplementary', action='store_true')

    args = parser.parse_args()
    d = vars(args)
    experiment = None

    if args.parameter_search:
        # search best parameter set with grid search
        from comet_ml import Optimizer
        config_base_path = "../configs/cometml/experiment.json"
        config_model_path = f"../configs/parameter_search/{args.net}.json"
        config_base = json.load(open(config_base_path))
        config_model = json.load(open(config_model_path))
        params_to_search = list(config_model.keys())
        config_base["parameters"] = {}  # initialize
        for param, v in config_model.items():
            if isinstance(v[0], str):
                config_base["parameters"][param] = {"type": "categorical", "values": v}
            else:
                config_base["parameters"][param] = {"type": "discrete", "values": v}

        opt = Optimizer(config=config_base, api_key=os.environ.get("API_KEY"))
        for experiment in opt.get_experiments(
                project_name=os.environ.get("PROJECT"), workspace=os.environ.get("WORKSPACE"), log_code=False):
            # replacement
            for param in params_to_search:
                d[param] = experiment.get_parameter(param)

            main(args, experiment)
            experiment.end()
    else:
        if args.cometml:
            from comet_ml import Experiment
            # Create an experiment with your api key
            experiment = Experiment(
                api_key=os.environ.get("API_KEY"),
                project_name=os.environ.get("PROJECT"),
                workspace=os.environ.get("WORKSPACE"),
            )
        # if there exist, you can use best parameter set.
        try:
            best_params_dir = "../configs/best_params"
            # if args.train_rate == 0.6:
            #     split_type = "supervised"
            # elif args.train_rate == 0.025:
            #     split_type = "semisupervised"
            split_type = "supervised"
            best_params_file = f"{best_params_dir}/best_params_{split_type}"

            # if "scalability" in args.exp or args.conditional:
            #     df = pd.read_csv(f"{best_params_file}_conditional.csv")
            if args.full_search_params:
                if not args.supplementary:
                    #  use the best parameters that maximized f1-macro
                    df = pd.read_csv(
                        f"{best_params_dir}/full_hyperparameter_search/best_params_supervised_f1macro.csv")

                else:
                    #  use the best parameters that maximized accuracy
                    print("supplementary experiment")
                    df = pd.read_csv(
                        f"{best_params_dir}/full_hyperparameter_search/best_params_supervised_accuracy.csv")

                df = df.query("exp == @args.exp")
            else:
                df = pd.read_csv(f"{best_params_file}.csv")

            if args.full_search_params:
                # convert GenCAT_cora_*_[012] to GenCAT_cora_*_0
                dataset = args.dataset[:-1] + "0"

            else:
                dataset = args.dataset.split("_")[1] if "GenCAT" in args.dataset else args.dataset
            net = args.net.split("-")[1] if "JK-" in args.net else args.net
            print(dataset)

            if ("GenCAT_cora_60000_100000" in args.dataset) or (
                    "GenCAT_cora_120000_200000" in args.dataset):
                print("use best parameters tuned for GenCAT_cora_15000_25000_0 dataset")
                df = df.query(
                    "net == @args.net & dataset == 'GenCAT_cora_15000_25000_0'").reset_index()
            else:
                df = df.query("net == @args.net & dataset == @dataset").reset_index()

            if not df.empty:
                print("best params found!")
                config_model_path = f"../configs/parameter_search/{net}.json"
                config_model = json.load(open(config_model_path))
                # replacement
                for k in config_model.keys():
                    if isinstance(d[k], int):
                        d[k] = int(df.at[0, k])
                    elif isinstance(d[k], float):
                        d[k] = float(df.at[0, k])
                    else:
                        d[k] = df.at[0, k]
                if "JK" in args.net:
                    print("Jumping Knowledge")
                    d["JK"] = df.at[0, "JK"]  # model with jumping knowldge
            else:
                print("best params not found")
        except FileNotFoundError as e:
            print(e)
            pass

        print(args)
        main(args, experiment)
