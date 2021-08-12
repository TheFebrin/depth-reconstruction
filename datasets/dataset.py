import os
import sys
import inspect
import torch
from typing import Union
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
import cv2

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(currentdir)
sys.path.append(parentdir)

from utils.util_functions import get_normals_from_depth


class DepthDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        color_path: str,
        depth_path: str,
        random_vertical_flip: bool,
        random_horizontal_flip: bool,
        image_transform=None,
        depth_transform=None,
        common_transform=None,
    ):

        self._csv_path = csv_path
        self._color_path = color_path
        self._depth_path = depth_path
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.common_transform = common_transform
        self.df = pd.read_csv(csv_path)
        self._random_vertical_flip = random_vertical_flip
        self._random_horizontal_flip = random_horizontal_flip

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx][1]

        color_img = Image.open(os.path.join(self._color_path, img_name))
        depth_img = Image.open(os.path.join(self._depth_path, img_name))

        if self._random_horizontal_flip and np.random.rand() < 0.5:
            color_img = TF.hflip(color_img)
            depth_img = TF.hflip(depth_img)

        if self._random_vertical_flip and np.random.rand() < 0.5:
            color_img = TF.vflip(color_img)
            depth_img = TF.vflip(depth_img)

        if self.common_transform:
            color_img = self.common_transform(color_img)
            depth_img = self.common_transform(depth_img)

        if self.image_transform:
            color_img = self.image_transform(color_img)

        if self.depth_transform:
            depth_img = self.depth_transform(depth_img)

        return color_img, depth_img


class NormalsDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        color_path: str,
        depth_path: str,
        common_albumentation_transform=None,
        color_albumentation_transform=None,
    ):
        self._csv_path = csv_path
        self._color_path = color_path
        self._depth_path = depth_path
        self.df = pd.read_csv(csv_path)
        self.common_albumentation_transform = common_albumentation_transform
        self.color_albumentation_transform = color_albumentation_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        color_img_name = self.df.iloc[idx][1]

        # As normals are n x m x 3 matices with values [-1, 1] they cannot be converted
        # to PIL.Image to use transforms.Resize
        depth_img_name = self.df.iloc[idx][2].replace('npz', 'png')

        # Read an image with OpenCV and convert it to the RGB colorspace
        color_img = cv2.imread(os.path.join(self._color_path, color_img_name))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        depth_img = cv2.imread(os.path.join(self._depth_path, depth_img_name), cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype(np.float32)

        # f, axes = plt.subplots(2, 2)
        # f.set_figheight(10)
        # f.set_figwidth(15)
        # axes[0, 0].imshow(color_img)
        # axes[0, 1].imshow(depth_img)

        if self.common_albumentation_transform:
            # usually RandomCrop
            transformed = self.common_albumentation_transform(image=color_img, mask=depth_img)
            color_img = transformed["image"]
            depth_img = transformed["mask"]

        if self.color_albumentation_transform:
            transformed = self.color_albumentation_transform(image=color_img)
            color_img = transformed["image"]

        color_img = torch.from_numpy(color_img.astype(np.float32))

        # axes[1, 0].imshow(color_img)
        # axes[1, 1].imshow(depth_img.squeeze())
        # plt.show()

        normals = torch.tensor(
            get_normals_from_depth(depth_img_arr=np.array(depth_img, dtype=np.float32).squeeze()),
            dtype=torch.float32
        )

        return color_img.permute(2, 0, 1), normals.permute(2, 0, 1)


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


def create_normals_dataset(
        csv_path: str,
        color_path: str,
        depth_path: str
) -> NormalsDataset:
    dataset = NormalsDataset(
        csv_path=csv_path,
        color_path=color_path,
        depth_path=depth_path,
        common_albumentation_transform=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomCrop(width=1720, height=980),
            A.Resize(width=1280, height=720)
        ]),
        color_albumentation_transform=A.Compose([
            A.ColorJitter(
                brightness=0.2, contrast=0.2,  saturation=0.2, hue=0.2, always_apply=False, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5
            ),
            A.GaussianBlur(
                blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5
            ),
        ]),
    )
    return dataset


def create_normals_dataset_test(
        csv_path: str,
        color_path: str,
        depth_path: str
) -> NormalsDataset:
    dataset = NormalsDataset(
        csv_path=csv_path,
        color_path=color_path,
        depth_path=depth_path,
        common_albumentation_transform=A.Compose([
            A.Resize(width=1280, height=720)
        ]),
    )
    return dataset


def create_dataset(
        csv_path: str,
        color_path: str,
        depth_path: str,
        mode: str='train',
) -> Union[NormalsDataset, DepthDataset]:
    assert mode in ('train', 'test'), f'Wrong mode: {mode}'
    if mode == 'train':
        dataset_f = create_normals_dataset
    else:
        dataset_f = create_normals_dataset_test

    return dataset_f(
        csv_path=csv_path,
        color_path=color_path,
        depth_path=depth_path
    )


def test_normals_dataset():
    dataset = create_normals_dataset(
        csv_path='datasets/color_normals_test_df.csv',
        color_path='color',
        depth_path='depth',
    )

    # Test production dataset
    # dataset = create_normals_dataset_test(
    #     csv_path='datasets/color_production_normals_test_df.csv',
    #     color_path='color_production',
    #     depth_path='depth_production',
    # )

    print(len(dataset))

    for x, y in dataset:
        print(x.shape, y.shape)
        break

    dataloader = DataLoader(
        dataset, batch_size=1,
        shuffle=True, num_workers=0
    )

    print('\n=============================\n')

    for x, y in dataloader:
        print(x.shape, y.shape)

        print('Color:')
        color_img = x[0].permute(1, 2, 0).numpy()
        print(color_img.min(), color_img.max(), color_img.shape, color_img.dtype, type(color_img))
        plt.figure(figsize=(15, 10))
        plt.imshow(color_img.astype(np.uint8))
        plt.show()

        print('Normals:')
        normals_img = y[0].permute(1, 2, 0).numpy()
        print(normals_img.min(), normals_img.max(), normals_img.shape, normals_img.dtype, type(normals_img))
        plt.figure(figsize=(15, 10))
        plt.imshow(normals_img)
        plt.show()
        break


def test_depth_dataset():
    dataset = create_normals_dataset(
        csv_path='datasets/color_normals_test_df.csv',
        color_path='color',
        depth_path='depth',
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
        print('Color img:')
        print(color_img.min(), color_img.max(), color_img.shape, color_img.dtype, type(color_img))
        plt.imshow(color_img)
        plt.show()

        depth_img = y[0].permute(1, 2, 0).numpy()
        print('Depth img:')
        print(depth_img.min(), depth_img.max(), depth_img.shape, depth_img.dtype, type(depth_img))
        plt.imshow(depth_img)
        plt.show()
        break


if __name__ == '__main__':
    # current_dir = os.getcwd()
    test_normals_dataset()
    # print('\n\n === END === \n\n')
    # test_depth_dataset()
