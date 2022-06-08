"""
Run this file if you just want to download a dataset locally.
"""
import argparse


def load_dataset(dataset):
    from models.dataset_utils import DataLoader

    _ = DataLoader(dataset, data_dir="./data/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Cora')
    args = parser.parse_args()

    load_dataset(args.dataset)
