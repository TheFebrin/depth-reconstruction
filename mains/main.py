"""
Example:

python main.py --epochs 5 --batch-size 4 --learning-rate 0.0001 --unet-downsample 6 --train-csv ../datasets/color_normals_train_df.csv \
    --train-color-path ../color --train-depth-path ../depth --device cuda:5 --name synthetic-training-1

python main.py --epochs 5 --batch-size 4 --learning-rate 0.0001 --unet-downsample 6  --train-csv ../datasets/color_production_normals_train_df.csv \
    --train-color-path ../color_production --train-depth-path ../depth_production --device cuda:6 --name production-training-1
"""
from comet_ml import Experiment, ConfusionMatrix

import sys, os, inspect
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm import tqdm

torch.cuda.empty_cache()

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(currentdir)
sys.path.append(parentdir)
sys.path.append(parentdir + '/models')

from models.unet import UNet
from datasets.dataset import NormalsDataset


experiment = Experiment(
    api_key=os.environ['COMET_ML_API_KEY'],
    project_name="depth-reconstruction",
    workspace="thefebrin",
)
experiment.add_tag('pytorch-depth-reconstruction')


def save_model():
    pass

def load_model():
    pass

def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for training.'
    )
    parser.add_argument(
        '--name', required=False, type=str, help='Name passed to CometML'
    )
    parser.add_argument(
        '--epochs', required=True, type=int, help='How many epochs.'
    )
    parser.add_argument(
        '--batch-size', required=True, type=int, default=1, help='Batch size.'
    )
    parser.add_argument(
        '--learning-rate', required=True, type=float, help='Model learning rate.'
    )
    parser.add_argument(
        '--unet-downsample', required=True, type=int, default=6, help='Unet downsample param.'
    )
    parser.add_argument(
        '--train-csv', required=True, type=str, help='Path to csv file with training file names.'
    )
    parser.add_argument(
        '--train-color-path', required=True, type=str, help='Path to folders with color train data.'
    )
    parser.add_argument(
        '--train-depth-path', required=True, type=str, help='Path to folders with depth train data.'
    )
    parser.add_argument(
        '--device', required=True, type=str, default='cuda:7', help='Which GPU.'
    )
    return parser.parse_args()

def create_dataset(csv_path: str, color_path: str, depth_path: str) -> NormalsDataset:
    dataset = NormalsDataset(
        csv_path=csv_path,
        color_path=color_path,
        depth_path=depth_path,
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
    return dataset


def train(
    model,
    train_dataloader,
    test_synthetic_dataloader,
    test_production_dataloader,
    epochs: int,
    criterion,
    optimizer: torch.optim.Optimizer,
    calculate_valid_frequency=1000,  # calculate model performance on valid set every 1000 steps
    device: str=None,
):
    print('STARTING TRAINING')
    print(f'train_dataloader: {len(train_dataloader)} | test_synthetic_dataloader: {len(test_synthetic_dataloader)} | test_production_dataloader: {len(test_production_dataloader)}')

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

    with experiment.train():
        # print("[Comet ML] Log histogram of model weights. May take a few minutes.")
        # weights = []
        # for name in model.named_parameters():
        #     if 'weight' in name[0]:
        #         weights += name[1].detach().numpy().tolist()
        # experiment.log_histogram_3d(weights, step=0)

        model = model.to(device)
        model.train()

        total_steps: int = 0
        for epoch in range(epochs):
            experiment.log_current_epoch(epoch)
            print(f'Epoch: {epoch + 1} / {epochs}')

            epoch_loss: float = 0.0
            steps: int = 0

            for x, y in tqdm(train_dataloader, desc=f'Epoch: {epoch + 1} / {epochs} | Total steps: {total_steps}'):
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = criterion(pred, y)

                experiment.log_metric("one_batch_loss", loss, step=total_steps)
                epoch_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (total_steps + 1) % calculate_valid_frequency == 0:
                    model.eval()
                    with torch.no_grad():
                        # save images
                        experiment.log_image(x[0].detach().cpu().permute(1, 2, 0), name=f'color_image_step_{steps}')
                        experiment.log_image(y[0].detach().cpu().permute(1, 2, 0), name=f'normals_image_step_{steps}')
                        experiment.log_image(pred[0].detach().cpu().permute(1, 2, 0), name=f'prediction_image_step_{steps}')

                        # save mean loss
                        experiment.log_metric(f'mean_loss_after_step_{steps}', epoch_loss / steps, step=total_steps)

                        for x, y in tqdm(test_synthetic_dataloader, desc=f'Validating on synthetic test set. Epoch: {epoch + 1} | Total steps: {total_steps}.'):
                            x = x.to(device)
                            y = y.to(device)
                            pred = model(x)
                            valid_loss = criterion(pred, y)
                            experiment.log_metric(f'valid_synthetic_loss_epoch_{epoch + 1}', valid_loss, step=total_steps)

                        for x, y in tqdm(test_production_dataloader, desc=f'Validating on production test set. Epoch: {epoch + 1} | Total steps: {total_steps}.'):
                            x = x.to(device)
                            y = y.to(device)
                            pred = model(x)
                            valid_loss = criterion(pred, y)
                            experiment.log_metric(f'valid_production_loss_epoch_{epoch + 1}', valid_loss, step=total_steps)

                    model.train()

                steps += 1
                total_steps += 1

            # End of epoch
            print(f'Mean epoch {epoch + 1} loss: {epoch_loss / steps}')
            experiment.log_metric(f'mean_loss_epoch_{epoch + 1}', epoch_loss / steps, step=total_steps)


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
    args              = parse_args()
    name              = args.name
    epochs            = args.epochs
    batch_size        = args.batch_size
    learning_rate     = args.batch_size
    unet_downsample   = args.unet_downsample
    device            = args.device
    train_csv         = args.train_csv
    train_color_path  = args.train_color_path
    train_depth_path  = args.train_depth_path

    model = UNet(downsample=unet_downsample, in_channels=3, out_channels=3)
    # model.half()
    optimizer = torch.optim.Adam(
        lr=learning_rate,
        betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
        params=model.parameters()
    )
    """
    Use SGD and be the king of the word!
        * https://arxiv.org/pdf/1812.01187.pdf - practical - changes life
        * https://arxiv.org/pdf/1506.01186.pdf - some critique 
        * https://arxiv.org/pdf/1711.04623.pdf
        * http://proceedings.mlr.press/v70/arpit17a/arpit17a.pdf
        
    * Start with Adam with weight_decay (output has to be 0 - 1) + momentum + AMS grad
    * add LR scheduler 
    """
    criterion = torch.nn.MSELoss()  # TODO: Consider SmoothL1 https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
    test_dataset = create_dataset(
        csv_path='../datasets/color_normals_test_df.csv',
        color_path='../color',
        depth_path='../depth',
    )
    test_production_dataset = create_dataset(
        csv_path='../datasets/color_production_normals_test_df.csv',
        color_path='../color_production',
        depth_path='../depth_production',
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
        unet_downsample: unet_downsample,
        epochs: epochs,
        batch_size: batch_size,
        learning_rate: learning_rate,
    })
    experiment.set_model_graph(str(model))

    # for x, y in tqdm(dataloader):
    #     print(x.shape)
    #     plt.imshow(np.dstack(x[0].numpy()))
    #     print(y[0])
    #     print('MADXXXL ', y[0].max())
    #     print(x[0].mean(), x[0].std())
    #     print(x[0])
    #     plt.show()
    #     break

    train(
        model=model,
        train_dataloader=train_dataloader,
        test_synthetic_dataloader=test_synthetic_dataloader,
        test_production_dataloader=test_production_dataloader,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
