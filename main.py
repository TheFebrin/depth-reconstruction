"""
Example:

1. Local debug  on cpu:
python main.py --device cpu --name debug

2. GPU
python main.py --device cuda:5 --name synthetic-training-1
"""

from comet_ml import Experiment, ConfusionMatrix

import sys
import os
import torch
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm import tqdm
from typing import Union
from models.unet_epfl import UNet as UNetEPFL
from models.unet import UNet
from datasets.dataset import NormalsDataset, DepthDataset, create_dataset
from trainers.train import train

torch.cuda.empty_cache()

experiment = Experiment(
    api_key=os.environ['COMET_ML_API_KEY'],
    project_name="depth-reconstruction",
    workspace="thefebrin",
)
experiment.add_tag('pytorch-depth-reconstruction')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for training.'
    )
    parser.add_argument(
        '--name', required=False, type=str, help='Name passed to CometML'
    )
    parser.add_argument(
        '--device', required=True, type=str, default='cuda:7', help='Which GPU.'
    )
    return parser.parse_args()


def main() -> int:
    config = yaml.safe_load(open("config.yaml"))

    args              = parse_args()
    name              = args.name
    device            = args.device

    epochs            = config['EPOCHS']
    batch_size        = config['BATCH_SIZE']
    learning_rate     = config['LEARNING_RATE']
    train_csv         = config['TRAIN_CSV_PATH']
    train_color_path  = config['TRAIN_COLOR_IMAGES_PATH']
    train_depth_path  = config['TRAIN_DEPTH_IMAGES_PATH']
    optimizer_name    = config['OPTIMIZER']
    criterion_name    = config['CRITERION']
    validate_every    = config['VALIDATE_EVERY']

    model = UNet(n_channels=3, n_classes=3, bilinear=False)

    assert optimizer_name in ('adam', 'sgd'), f'Wrong optimizer name: {optimizer_name}'
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            lr=0.0001,
            betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
            params=model.parameters()
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            lr=learning_rate,
            momentum=0.9,
            params=model.parameters()
        )

    assert criterion_name in ('mse', 'smoothl1'), f'Wrong criterion name: {optimizer_name}'
    if criterion_name == 'mse':
        criterion = torch.nn.MSELoss()
    elif criterion_name == 'smoothl1':
        # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
        criterion = torch.nn.SmoothL1Loss()

    test_dataset = create_dataset(
        csv_path='datasets/color_normals_test_df.csv',
        color_path='color',
        depth_path='depth',
        mode='test',
    )
    test_production_dataset = create_dataset(
        csv_path='datasets/color_production_normals_test_df.csv',
        color_path='color_production',
        depth_path='depth_production',
        mode='test',
    )
    train_dataset = create_dataset(
        csv_path=train_csv,
        color_path=train_color_path,
        depth_path=train_depth_path,
        mode='train',
    )

    test_synthetic_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
    )
    test_production_dataloader = DataLoader(
        test_production_dataset, batch_size=batch_size, shuffle=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )

    # Log hyperparameters to Comet
    experiment.set_name(name)
    experiment.add_tag(name)
    experiment.log_parameters({
        epochs: epochs,
        batch_size: batch_size,
        learning_rate: learning_rate,
    })
    experiment.set_model_graph(str(model))

    train(
        model=model,
        train_dataloader=train_dataloader,
        test_synthetic_dataloader=test_synthetic_dataloader,
        test_production_dataloader=test_production_dataloader,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        experiment=experiment,
        calculate_valid_frequency=validate_every,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
