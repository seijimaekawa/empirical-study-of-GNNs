# Beyond Real-world Benchmark Datasets: An Empirical Study of Node Classification with GNNs

This paper is accepted to NeurIPS 2022 Datasets and Benchmarks🎉🎉　 [[paper](https://openreview.net/forum?id=bSULxOy3On)]

Our empirical studies clarify the strengths and weaknesses of GNNs from four major characteristics of real-world graphs with class labels of nodes, i.e., 1) class size distributions (balanced vs. imbalanced), 2) edge connection proportions between classes (homophilic vs. heterophilic), 3) attribute values (biased vs. random), and 4) graph sizes (small vs. large).

# Supported Models

MLP, [GCN](https://github.com/tkipf/pygcn), [ChebNet](https://arxiv.org/abs/1606.09375), [MoNet](https://arxiv.org/abs/1611.08402), [GAT](https://github.com/PetarV-/GAT),  [SGC](https://arxiv.org/abs/1902.07153), [JK-GCN](https://arxiv.org/abs/1806.03536), [JK-GAT](https://arxiv.org/abs/1806.03536), [JK-GraphSAGE](https://arxiv.org/abs/1806.03536), [GraphSAGE](http://snap.stanford.edu/graphsage/), [GraphSAINT-GAT](https://arxiv.org/abs/1907.04931), [GraphSAINT-GraphSAGE](https://arxiv.org/abs/1907.04931), [Shadow-GAT](https://github.com/facebookresearch/shaDow_GNN), [Shadow-GraphSAGE](https://github.com/facebookresearch/shaDow_GNN), [H2GCN](https://arxiv.org/abs/2006.11468), [FSGNN](https://arxiv.org/abs/2105.07634), [GPRGNN](https://github.com/jianhao2016/GPRGNN), [LINKX](https://github.com/cuai/non-homophily-large-scale)

# Installation
All our experiments are executed with Python3.7.13. 
Please run scripts below to prepare an environment for our codebase. 

`pip install torch torchvision torchaudio`

Please see the [official instruction](https://pytorch.org/get-started/locally/) to install pytorch. We use torch==1.12.1. 

`pip install -r requirements.txt`

# Dataset Generation ([GenCAT](https://arxiv.org/abs/2109.04639))

Choose a base dataset and Generate dataset with GenCAT. If the base dataset is not in your directory, it will be downloaded automatically.
Dataset generated with pre-set parameters will be saved under the `data` directory.

```
python scripts/run_gencat.py --dataset cora

# To reproduce sythetic datasets for Section 6.1.1 (various class size distributions)
python scripts/run_gencat.py --dataset cora --exp classsize
 
# To reproduce sythetic datasets for Section 6.1.2 (various edge connection proportions between classes)
python scripts/run_gencat.py --dataset cora --exp hetero_homo

# To reproduce sythetic datasets for Section 6.1.3 (various attributes)
python scripts/run_gencat.py --dataset cora --exp attribute

# To reproduce sythetic datasets for Section 6.1.4 (various numbers of nodes and edges)
python scripts/run_gencat.py --dataset cora --exp scalability_node_edge

# To reproduce sythetic datasets for Section 6.2 (various numbers of edges)
python scripts/run_gencat.py --dataset cora --exp scalability_edge
```

### [Optional] Already Generated Dataset Link
If you do not want to generate, you can download synthetic datasets that we use in the paper: 
[Google Drive](https://drive.google.com/file/d/1JedynF0F-JJgCFBS4CYF-DqPZXuOcJSh/view?usp=sharing) or [Our Lab Repository](http://www-bigdata.ist.osaka-u.ac.jp/ja/download?name=Experiment_Data_zip) (These links provide the same dataset). 

Please put the unzipped folder as `./data/` after downloading it.

# Reproduction of Experiments in the Paper
All plots in Figure 1-6 are shown in a [notebook](https://github.com/seijimaekawa/empirical-study-of-GNNs/blob/main/notebooks/final_plots.ipynb). 

The raw experimental results are stored in [csv-formated files](https://github.com/seijimaekawa/empirical-study-of-GNNs/tree/main/notebooks/final_results). 

## Experiments in Supplementary Material
All plots are shown in a [notebook](https://github.com/seijimaekawa/empirical-study-of-GNNs/blob/main/notebooks/supplementary.ipynb). 

You can find the raw experimental results in [csv-formated files](https://github.com/seijimaekawa/empirical-study-of-GNNs/tree/main/notebooks/supplementary).

## Hyperparameters
### Search Space
The hyperparameter search space for each model is listed in [json files](https://github.com/seijimaekawa/empirical-study-of-GNNs/tree/main/configs/parameter_search).
### The Best Sets of Hyperparameters for Each Experiment
Also, we show the [best parameter sets](https://github.com/seijimaekawa/empirical-study-of-GNNs/tree/main/configs/best_params/full_hyperparameter_search) used for the experiments.

# Instruction for Running GNNs

If the base dataset is not in your directory, it will be downloaded automatically.
Please go to folder `models`

```
python train_model.py --train_rate 0.6 --val_rate 0.2 --RPMAX 2 --dataset cora --net GCN
```

# Format of Datasets

Although all datasets are internally converted to the pytorch format, you can convert datasets into other formats so that you can use datasets for your own use cases. The converted data will be stored in the dataset's directory. Formats that this codebase supports are npz, [semb](https://github.com/gemslab/strucEmbedding-graphlibrary), and [planetoid](https://github.com/kimiyoung/planetoid).

An example script is following:
```
python scripts/converter.py --format npz --dataset GenCAT_texas
```

# Customization

Below we describe how to customize this codebase for your own research / product.

## How to Support Your Own GNN models?

Add your model class in `./models/GNN_models.py`. You would also need to do some minor update to **net_dict** variable, **import from GNN_models** and **parser** of the net argument in `./models/train_model.py`, so that you can specify that model with an argument.

## How to Prepare Your Own Dataset?

Add your dataset class in `./models/dataset_utils.py`. You would also need to do some minor update to **DataLoader** function in `./models/dataset_utils.py` so that you can use the class. The raw data can be in any format, but after preprocessing with the dataset class, it needs to be converted to pytorch format. When you use GenCAT to create your datasets, you would also need to add your dataset into **datasets_to_convert** list in `./scripts/run_gencat.py` so that the format will be converted to the planetoid format.

## How to Tune Hyperparameters?

We use third party service, [cometml](https://www.comet.ml/site/), to tune hyperparameters with a grid search algorithm. If you choose the same way, follow the instructions below.

1. Set all parameters to **parser** in `./models/train_model.py`.
2. Write all the parameters to be explored into a json file({YOUE_MODEL}.json) and set it in `./configs/parameter_search/`.
3. Set your cometml information as environment variables.
4. Run `./models/train_model.py` with `--parameter_search`.

```
python train_model.py --train_rate 0.6 --val_rate 0.2 --RPMAX 10 --dataset cora --net GCN --parameter_search
```

A simple way to use the best parameter set you explored is to add it in `./configs/best_params/best_params_supervised.csv` as a new row. If there is a line in the file with a matching dataset/net combination, then you can run `./models/train_model.py` using the best parameters. In this case, you do not have to set the best parameters as arguments.
You can also choose to set the best parameters as arguments when running the code.

# Built-in Datasets

This framework allows users to use real-world datasets as follows:
  | Dataset                                                 | # Nodes | # Edges |
  | ------------------------------------------------------- | ------- | ------- |
  | [Cora](https://github.com/kimiyoung/planetoid)          | 2,708   | 5,278   |
  | [Pubmed](https://github.com/kimiyoung/planetoid)        | 19,717  | 44,324  |
  | [Citeseer](https://github.com/kimiyoung/planetoid)      | 3,327   | 4,552   |
  | [Texas](https://openreview.net/forum?id=S1e2agrFvS)     | 183     | 295     |
  | [Wisconsin](https://openreview.net/forum?id=S1e2agrFvS) | 251     | 466     |
  | [Cornell](https://openreview.net/forum?id=S1e2agrFvS)   | 183     | 280     |
  | [Actor](https://openreview.net/forum?id=S1e2agrFvS)     | 7,600   | 26,752  |
  | [Chameleon](https://arxiv.org/abs/1909.13021)           | 2,277   | 31,421  |
  | [Squirrel](https://arxiv.org/abs/1909.13021)            | 5,200   | 198,493 |
  | [BlogCatalog](http://snap.stanford.edu/node2vec/)       | 5,196   | 343,486 |
  | [Flickr](https://arxiv.org/abs/2009.00826)              | 7,575   | 479,476 |

By changing `--dataset [dataset name]`, users can choose a dataset. 
