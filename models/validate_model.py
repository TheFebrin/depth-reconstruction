import os
import sys
import inspect
import torch
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(currentdir)
sys.path.append(parentdir)

from datasets.dataset import NormalsDataset
from models.unet import UNet


def load_production_dataset():
    return NormalsDataset(
        csv_path='datasets/color_production_normals_test_df.csv',
        color_path='color_production',
        depth_path='depth_production',
        random_vertical_flip=True,
        random_horizontal_flip=True,
        common_transform=transforms.Compose([
            transforms.Resize((128, 256)),
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

def load_model(path):
    model = UNet(n_channels=3, n_classes=3, bilinear=False)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

def main():
    dataset = load_production_dataset()
    synthetic_model = load_model(path='models/saved_models/best_synthetic_model.pth')
    production_model = load_model(path='models/saved_models/best_production_model.pth')

    n_rows = 5

    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 15))

    for ax, col in zip(axes[0], ['Input image', 'Ground truth', 'Synthetic model', 'Production model']):
        ax.set_title(col)

    for i, D in enumerate(dataset):
        if i == n_rows:
            break

        x, y = D

        color_img = x.permute(1, 2, 0).numpy()
        normals_img = y.permute(1, 2, 0).numpy()
        synthetic_pred = synthetic_model(x.view(-1, *x.shape))[0]
        production_pred = production_model(x.view(-1, *x.shape))[0]

        axes[i][0].imshow(color_img)
        axes[i][1].imshow(normals_img)
        axes[i][2].imshow(synthetic_pred.detach().permute(1, 2, 0).numpy())
        axes[i][3].imshow(production_pred.detach().permute(1, 2, 0).numpy())

    plt.show()

if __name__ == '__main__':
    main()