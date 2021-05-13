import os
import torch
from torchvision import transforms, utils
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


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


if __name__ == '__main__':
    # current_dir = os.getcwd()

    dataset = DepthDataset(
        color_images_path='../color/',
        depth_images_path='../depth/',
        common_transform=transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
        ])
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
        break

    test, train = torch.utils.data.random_split(dataset, [1000, 9000])
    print(test, train)