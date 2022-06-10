# Beyond Real-world Benchmark Datasets: An Empirical Study of Node Classification with GNNs

# Supported Models

[GCN](https://github.com/tkipf/pygcn), MLP, [GAT](https://github.com/PetarV-/GAT), [GraphSAGE](http://snap.stanford.edu/graphsage/), [ChebNet](https://arxiv.org/abs/1606.09375), [SGC](https://arxiv.org/abs/1902.07153), [H2GCN](https://arxiv.org/abs/2006.11468), [MoNet](https://arxiv.org/abs/1611.08402), [GPRGNN](https://github.com/jianhao2016/GPRGNN), [FSGNN](https://arxiv.org/abs/2105.07634), [JK-GCN](https://arxiv.org/abs/1806.03536),[JK-GAT](https://arxiv.org/abs/1806.03536), [JK-GraphSAGE](https://arxiv.org/abs/1806.03536), [GraphSAINT-GAT](https://arxiv.org/abs/1907.04931), [GraphSAINT-GraphSAGE](https://arxiv.org/abs/1907.04931), [Shadow-GAT](https://github.com/facebookresearch/shaDow_GNN), [Shadow-GraphSAGE](https://github.com/facebookresearch/shaDow_GNN)

# Dataset Generation (GenCAT)

Choose a base dataset and Generate dataset with GenCAT. If the base dataset is not in your directory, it will be downloaded automatically.
Dataset generated with pre-set parameters will be saved under the `data` directory.

```
python scripts/run_gencat.py --dataset cora
```

You can download synthetic datasets that we use in the paper: 
[dataset link](https://drive.google.com/file/d/1B7X65BoPij8sEmL491T-LDlzrm5aATRH/view?usp=sharing)

Please put the unzipped folder as `./data/`.

# Reproduction of Experiments in the Paper
All plots in Figure 1-6 are shown in a [notebook](https://github.com/seijimaekawa/empirical-study-of-GNNs/blob/main/notebooks/final_plots.ipynb). 

The raw experimental results are stored in [csv-formated files](https://github.com/seijimaekawa/empirical-study-of-GNNs/tree/main/notebooks/final_results). 

## Hyperparameters
The hyperparameter search space for each model is listed in [json files](https://github.com/seijimaekawa/empirical-study-of-GNNs/tree/main/configs/parameter_search).

Also, we show the [best parameter sets](https://github.com/seijimaekawa/empirical-study-of-GNNs/tree/main/configs/best_params/full_hyperparameter_search) used for the experiments.

# Run GNN

If the base dataset is not in your directory, it will be downloaded automatically.
Please go to folder `models`

```
python train_model.py --train_rate 0.6 --val_rate 0.2 --RPMAX 2 --dataset cora --net GCN
```

# Format of Datasets

Although all the dataset are internally converted to pytorch format, you can convert the dataset format into another format so that you can use the datasets for your use case. The converted data will be stored in the dataset's directory. The selectable formats are npz, [semb](https://github.com/gemslab/strucEmbedding-graphlibrary), and [planetoid](https://github.com/kimiyoung/planetoid).

```
python scripts/converter.py --format npz --dataset GenCAT_texas
```

# Customization

Below we describe how to customize this code base for your own research / product.

## How to Support Your Own GNN model?

Add your model class in `./models/GNN_models.py`. You would also need to do some minor update to **net_dict** variable, **import from GNN_models** and **parser** of the net argument in `./models/train_model.py`, so that you can specify that model with an argument.

## How to Prepare Your Own Dataset?

Add your dataset class in `./models/dataset_utils.py`. You would also need to do some minor update to **DataLoader** function in `./models/dataset_utils.py` so that you can use the class. The raw data can be in any format, but after preprocessing with the dataset class, it needs to be converted to pytorch format. When you use GenCAT to your dataset, you would also need to add your dataset into **datasets_to_convert** list in `./scripts/run_gencat.py` so that the format will be converted to the planetoid format.

## How to Tune Hyperparameters?

We use third party service, cometml, to tune hyperparameters with grid search algorithm. If you choose the same way, follow the instructions below.

1. Set all parameters to **parser** in `./models/train_model.py`.
2. Write all the parameters to be explored into a json file({YOUE_MODEL}.json) and set it in `./configs/parameter_search/`.
3. Set your cometml information as environment variables.
4. Run `./models/train_model.py` with `--parameter_search`.

```
python train_model.py --train_rate 0.6 --val_rate 0.2 --RPMAX 10 --dataset cora --net GCN --parameter_search
```

One way to use the best parameter set you explored is to add it in `./configs/best_params/best_params_supervised.csv` as a new row. If there is a line in the file with a matching dataset/net combination, then you can run `./models/train_model.py` using the best parameters without setting the best parameters as arguments.
You can also choose to set the best parameters as arguments when running the code.

# Built-in Datasets

This framework allows users to use real-world datasets as follows:
  | Dataset                                                 | # Nodes | # Edges |
  | ------------------------------------------------------- | ------- | ------- |
  | [Cora](https://github.com/kimiyoung/planetoid)          | 2,708   | 5,278   |
  | [Citeseer](https://github.com/kimiyoung/planetoid)      | 3,327   | 4,552   |
  | [Pubmed](https://github.com/kimiyoung/planetoid)        | 19,717  | 44,324  |
  | [Cornell](https://openreview.net/forum?id=S1e2agrFvS)   | 183     | 277     |
  | [Texas](https://openreview.net/forum?id=S1e2agrFvS)     | 183     | 279     |
  | [Wisconsin](https://openreview.net/forum?id=S1e2agrFvS) | 251     | 499     |
  | [Actor](https://openreview.net/forum?id=S1e2agrFvS)     | 7,600   | 26,659  |
  | [Chameleon](https://arxiv.org/abs/1909.13021)           | 2,277   | 31,371  |
  | [Squirrel](https://arxiv.org/abs/1909.13021)            | 5,201   | 198,353 |
  | [BlogCatalog](http://snap.stanford.edu/node2vec/)       | 5,196   | 343,486 |
  | [Flickr](https://arxiv.org/abs/2009.00826)              | 7,575   | 479,476 |
  | [Wiki](https://github.com/GRAND-Lab/MGAE)               | 2,405   | 17,981  |

By changing `--dataset [dataset name]`, users can choose a dataset. 
