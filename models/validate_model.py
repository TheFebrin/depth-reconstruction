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
from utils.util_functions import get_normals_from_depth


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


def test_model_on_dataset():
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


def test_model_on_images():
    model_1_name = 'saved_models/best_synthetic_model.pth'
    model_2_name = 'saved_models/new_model_0_9000.pth'
    model_3_name = 'saved_models/new_model_1_12000.pth'


    model = load_model(path=model_1_name)
    model2 = load_model(path=model_2_name)
    model3 = load_model(path=model_3_name)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.tight_layout()

    img1 = Image.open('../images/source_right_blue.png')
    img1 = np.array(img1)
    axes[0][0].imshow(img1)
    img1 = torch.from_numpy(img1.astype(np.float32)).permute(2, 0, 1)

    img2 = Image.open('../images/source_right_white.png')
    img2 = np.array(img2)
    axes[1][0].imshow(img2)
    img2 = torch.from_numpy(img2.astype(np.float32)).permute(2, 0, 1)

    print('Start pred')
    with torch.no_grad():
        pred_img1 = model(img1.view(-1, *img1.shape))[0].detach().permute(1, 2, 0).numpy()
        print('Pred 1 / 4 done')
        pred_img2 = model(img2.view(-1, *img2.shape))[0].detach().permute(1, 2, 0).numpy()

        print('Pred 2 / 4')
        pred_img21 = model2(img1.view(-1, *img1.shape))[0].detach().permute(1, 2, 0).numpy()
        print('Pred 3 / 4')
        pred_img22 = model2(img2.view(-1, *img2.shape))[0].detach().permute(1, 2, 0).numpy()

        print('Pred 4 / 4')
        pred_img31 = model3(img1.view(-1, *img1.shape))[0].detach().permute(1, 2, 0).numpy()
        print('Pred 5 / 4')
        pred_img32 = model3(img2.view(-1, *img2.shape))[0].detach().permute(1, 2, 0).numpy()

        """
        print('Pred 6 / 4')
        pred_img41 = model4(img1.view(-1, *img1.shape))[0].detach().permute(1, 2, 0).numpy()
        print('Pred 7 / 4')
        pred_img42 = model4(img2.view(-1, *img2.shape))[0].detach().permute(1, 2, 0).numpy()
    """

    axes[0][0].imshow(pred_img1)
    axes[1][0].imshow(pred_img2)
    axes[0][0].set_title(model_1_name)
    axes[1][0].set_title(model_1_name)

    axes[0][1].imshow(pred_img21)
    axes[1][1].imshow(pred_img22)
    axes[0][1].set_title(model_2_name)
    axes[1][1].set_title(model_2_name)

    axes[0][2].imshow(pred_img31)
    axes[1][2].imshow(pred_img32)
    axes[0][2].set_title(model_3_name)
    axes[1][2].set_title(model_3_name)

    plt.show()


def test_one_model_vs_realsense():
        model_name = 'saved_models/new_model_3_33000.pth'
        # model_name = 'saved_models/best_synthetic_model.pth'
        # model_name = 'saved_models/best_model_14_144000.pth'

        model = load_model(path=model_name)

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.tight_layout()

        # img1 = Image.open('../images/production/color_production_1.png')
        # img1 = np.array(img1)
        # axes[0][0].imshow(img1)
        # img1 = torch.from_numpy(img1.astype(np.float32)).permute(2, 0, 1)
        # depth1 = np.array(Image.open('../images/production/depth_production_1.png'), dtype=np.float32)
        # axes[0][2].imshow(get_normals_from_depth(depth1))
        #
        # img2 = Image.open('../images/production/color_production_4.png')
        # img2 = np.array(img2)
        # axes[1][0].imshow(img2)
        # img2 = torch.from_numpy(img2.astype(np.float32)).permute(2, 0, 1)
        # depth2 = np.array(Image.open(
        #     '../images/production/depth_production_4.png'), dtype=np.float32)
        # axes[1][2].imshow(get_normals_from_depth(depth2))
        #
        # img3 = Image.open('../images/production/color_production_5.png')
        # img3 = np.array(img3)
        # axes[2][0].imshow(img3)
        # img3 = torch.from_numpy(img3.astype(np.float32)).permute(2, 0, 1)
        # depth3 = np.array(Image.open(
        #     '../images/production/depth_production_5.png'), dtype=np.float32)
        # axes[2][2].imshow(get_normals_from_depth(depth3))

        BASE_PATH = '../images/experiment_on_robot/'
        img1 = Image.open(BASE_PATH + 'item_left_images/item_left_height_1.png')
        img1 = np.array(img1)
        axes[0][0].imshow(img1)
        img1 = torch.from_numpy(img1.astype(np.float32)).permute(2, 0, 1)
        depth1 = np.array(
            Image.open(BASE_PATH + 'item_left_images/true_depth_item_left_height_1.png'), dtype=np.float32)
        axes[0][2].imshow(get_normals_from_depth(depth1))

        img2 = Image.open('../images/source_right_blue.png')
        img2 = np.array(img2)
        axes[1][0].imshow(img2)
        img2 = torch.from_numpy(img2.astype(np.float32)).permute(2, 0, 1)
        depth2 = np.array(Image.open(
            '../images/depth_source_right_blue.png'), dtype=np.float32)
        axes[1][2].imshow(get_normals_from_depth(depth2))

        # img3 = Image.open('../images/source_bot_white.png')
        # img3 = np.array(img3)
        # axes[2][0].imshow(img3)
        # img3 = torch.from_numpy(img3.astype(np.float32)).permute(2, 0, 1)
        # depth3 = np.array(Image.open(
        #     '../images/depth_source_bot_white.png'), dtype=np.float32)
        # axes[2][2].imshow(get_normals_from_depth(depth3))

        img3 = Image.open(BASE_PATH + 'item_top_images/item_top_height_1.png')
        img3 = np.array(img3)
        axes[2][0].imshow(img3)
        img3 = torch.from_numpy(img3.astype(np.float32)).permute(2, 0, 1)
        depth3 = np.array(
            Image.open(BASE_PATH + 'item_top_images/true_depth_item_top_height_1.png'), dtype=np.float32)
        axes[2][2].imshow(get_normals_from_depth(depth3))

        print('Start pred')
        with torch.no_grad():
            pred_img1 = model(img1.view(-1, *img1.shape))[0].detach().permute(1, 2, 0).numpy()
            print('Pred 1 / 4 done')
            pred_img2 = model(img2.view(-1, *img2.shape))[0].detach().permute(1, 2, 0).numpy()
            print('Pred 2 / 4')
            pred_img3 = model(img3.view(-1, *img3.shape))[0].detach().permute(1, 2, 0).numpy()
            # print('Pred 3 / 4')
            # pred_img4 = model(img4.view(-1, *img4.shape))[0].detach().permute(1, 2, 0).numpy()

            # print('Pred 4 / 4')
            # pred_img31 = model3(img1.view(-1, *img1.shape))[0].detach().permute(1, 2, 0).numpy()
            # print('Pred 5 / 4')
            # pred_img32 = model3(img2.view(-1, *img2.shape))[0].detach().permute(1, 2, 0).numpy()


            """
            print('Pred 6 / 4')
            pred_img41 = model4(img1.view(-1, *img1.shape))[0].detach().permute(1, 2, 0).numpy()
            print('Pred 7 / 4')
            pred_img42 = model4(img2.view(-1, *img2.shape))[0].detach().permute(1, 2, 0).numpy()
        """

        axes[0][1].imshow(pred_img1)
        axes[1][1].imshow(pred_img2)
        axes[2][1].imshow(pred_img3)
        # axes[3][1].imshow(pred_img4)

        # axes[0][0].set_title('RGB production images')
        # axes[1][0].set_title('RGB image')
        # axes[2][0].set_title('RGB image')
        # axes[3][0].set_title('RGB image')

        # axes[0][1].set_title('Predictions from ...')
        # axes[1][1].set_title('RGB image')
        # axes[2][1].set_title('RGB image')
        # axes[3][1].set_title('RGB image')

        # axes[0][2].set_title('Normals from Realsense depth')
        # axes[1][0].set_title('RGB image')
        # axes[2][0].set_title('RGB image')
        # axes[3][0].set_title('RGB image')


        plt.show()


def main():
	test_one_model_vs_realsense()


if __name__ == '__main__':
    main()
