import os
import sys
import inspect
import torch
from torchvision import transforms, utils
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

from utils.util_functions import get_normals_from_depth


class DepthDataset(Dataset):

    def __init__(
        self,
        color_images_path: str,
        depth_images_path: str,
        image_transform=None,
        depth_transform=None,
        common_transform=None,
    ):
        color_names = list(sorted(os.listdir(f'{color_images_path}')))
        depth_names = list(sorted(os.listdir(f'{depth_images_path}')))

        self._color_images_path = color_images_path
        self._depth_images_path = depth_images_path

        self.dataset = list(zip(color_names, depth_names))
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.common_transform = common_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datapoint_name = self.dataset[idx][0]
        label_name = self.dataset[idx][1]

        datapoint = Image.open(f'{self._color_images_path}/{datapoint_name}')
        label = Image.open(f'{self._depth_images_path}/{label_name}')

        if self.common_transform:
            datapoint = self.common_transform(datapoint)
            label = self.common_transform(label)

        if self.image_transform:
            datapoint = self.image_transform(datapoint)

        if self.depth_transform:
            label = self.depth_transform(label)

        return datapoint, label


class NormalsDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        color_path: str,
        depth_path: str,
        image_transform=None,
        depth_transform=None,
        common_transform=None,
    ):
        self._csv_path = csv_path
        self._color_path = color_path
        self._depth_path = depth_path
        self.df = pd.read_csv(csv_path)
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.common_transform = common_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        color_img_name = self.df.iloc[idx][1]
        """
        Note:
            As normals are n x m x 3 matices with values [-1, 1] they cannot be converted to PIL.Image to use transforms.Resize
        """
        depth_img_name = self.df.iloc[idx][2].replace('npz', 'png')

        color_img = Image.open(os.path.join(self._color_path, color_img_name))
        depth_img = Image.open(os.path.join(self._depth_path, depth_img_name))

        if self.common_transform:
            color_img = self.common_transform(color_img)
            depth_img = self.common_transform(depth_img)

        if self.image_transform:
            color_img = self.image_transform(color_img)

        if self.depth_transform:
            depth_img = self.depth_transform(depth_img)

        normals = np.array(get_normals_from_depth(depth_img_arr=np.array(depth_img).squeeze()), dtype=np.float32)
        normals = torch.tensor(normals, dtype=torch.float32).permute(2, 0, 1)

        return color_img, normals


if __name__ == '__main__':
    # current_dir = os.getcwd()

    dataset = NormalsDataset(
        csv_path='color_normals_test_df.csv',
        color_path='../color',
        depth_path='../depth',
        common_transform=transforms.Compose([
            # transforms.Resize((4 * 256, 7 * 256)),
            transforms.Resize((1 * 256, 2 * 256)),  # temporary lower the resolution
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ]),
        image_transform=transforms.Compose([
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # mean and std from ImageNet
        ]),
        depth_transform=transforms.Compose([
            # lambda x: x / 2 ** 16  # normalize depth, no needed in case of normals
        ]),
    )

    print(len(dataset))

    for x, y in dataset:
        print(x.shape, y.shape)
        break

    dataloader = DataLoader(
        dataset, batch_size=4,
        shuffle=True, num_workers=0
    )

    print('\n=============================\n')

    for x, y in dataloader:
        print(x.shape, y.shape)

        color_img = x[0].permute(1, 2, 0).numpy()
        print(color_img.min(), color_img.max(), color_img.shape, color_img.dtype, type(color_img))
        plt.imshow(color_img)
        plt.show()

        normals_img = y[0].permute(1, 2, 0).numpy()
        print(normals_img.min(), normals_img.max(), normals_img.shape, normals_img.dtype, type(normals_img))
        plt.imshow(normals_img)
        plt.show()
        break
