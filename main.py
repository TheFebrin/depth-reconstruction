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
from datasets.dataset import NormalsDataset, DepthDataset
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

def create_depth_dataset(csv_path: str, color_path: str, depth_path: str) -> DepthDataset:
    return DepthDataset(
        csv_path=csv_path,
        color_path=color_path,
        depth_path=depth_path,
        random_vertical_flip=True,
        random_horizontal_flip=True,
        common_transform=transforms.Compose([
            # transforms.Resize((4 * 256, 7 * 256)),
            transforms.Resize((128, 256)),  # temporary lower the resolution
            transforms.ToTensor(),
        ]),
        image_transform=transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # mean and std from ImageNet
        ]),
        depth_transform=transforms.Compose([
            lambda x: x.float(),
            lambda x: x / 2 ** 16  # normalize depth, no needed in case of normals
        ]),
    )

def create_normals_dataset(csv_path: str, color_path: str, depth_path: str) -> NormalsDataset:
    dataset = NormalsDataset(
        csv_path=csv_path,
        color_path=color_path,
        depth_path=depth_path,
        random_vertical_flip=True,
        random_horizontal_flip=True,
        common_transform=transforms.Compose([
            # transforms.Resize((4 * 256, 7 * 256)),
            transforms.Resize((128, 256)),  # temporary lower the resolution
            lambda x: np.array(x),
            transforms.ToTensor(),
        ]),
        image_transform=transforms.Compose([
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # mean and std from ImageNet
        ]),
        depth_transform=transforms.Compose([
            # lambda x: x / 2 ** 16  # normalize depth, no needed in case of normals
        ]),
    )
    return dataset

def create_dataset(csv_path: str, color_path: str, depth_path: str) -> Union[NormalsDataset, DepthDataset]:
    dataset_f = create_normals_dataset
    return dataset_f(
        csv_path=csv_path,
        color_path=color_path,
        depth_path=depth_path
    )

def main() -> int:
    """
    TODO:
        * crop tote from image
        * predict normals instead of depth
            https://xiaolonw.github.io/papers/deep3d.pdf
        * freeze validation sets from real and synthetic data

        * Add argparse 
        * Data augmentation
            - ✔ flips horizontal + vertical
            - ✔️standarize data - use the same mean and std as in Imagenet
                https://forums.fast.ai/t/is-normalizing-the-input-image-by-imagenet-mean-and-std-really-necessary/51338
            - rotations - tricky because rotation changes shape, consider adding a black pad, so image is a big square
            - autoaugment

        https://github.com/pytorch/vision/blob/e35793a1a4000db1f9f99673437c514e24e65451/torchvision/models/detection/roi_heads.py#L45
        box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    """

    """
    Use SGD and be the king of the word!
        * https://arxiv.org/pdf/1812.01187.pdf - practical - changes life
        * https://arxiv.org/pdf/1506.01186.pdf - some critique 
        * https://arxiv.org/pdf/1711.04623.pdf
        * http://proceedings.mlr.press/v70/arpit17a/arpit17a.pdf

    * Start with Adam with weight_decay (output has to be 0 - 1) + momentum + AMS grad
    * add LR scheduler 
    """

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
    )
    test_production_dataset = create_dataset(
        csv_path='datasets/color_production_normals_test_df.csv',
        color_path='color_production',
        depth_path='depth_production',
    )
    train_dataset = create_dataset(
        csv_path=train_csv,
        color_path=train_color_path,
        depth_path=train_depth_path,
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
