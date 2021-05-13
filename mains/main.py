from comet_ml import Experiment, ConfusionMatrix

import sys, os, inspect
import torch
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
from dataloaders.dataloader import DepthDataset


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

def create_dataset() -> DepthDataset:
    dataset = DepthDataset(
        color_images_path='../color/',
        depth_images_path='../depth/',
        common_transform=transforms.Compose([
            transforms.Resize((4 * 256, 7 * 256)),
            lambda x: np.array(x, dtype=np.float32),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]),
        image_transform=transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # mean and std from ImageNet
        ]),
        depth_transform=transforms.Compose([
            lambda x: x / 2 ** 16  # normalize depth
        ]),
    )
    return dataset


def train(
    model,
    train_dataloader,
    test_dataloader,
    epochs: int,
    criterion,
    optimizer: torch.optim.Optimizer,
    calculate_valid_frequency=1000,  # calculate model performance on valid set every 1000 steps
    device: str=None,
):
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

                pred_depth = model(x)
                loss = criterion(pred_depth, y)
                experiment.log_metric("one_batch_loss", loss, step=total_steps)
                epoch_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if total_steps % calculate_valid_frequency == 0:
                    model.eval()
                    with torch.no_grad():
                        for x, y in tqdm(test_dataloader, desc=f'Validating on test set. Epoch: {epoch + 1} | Total steps: {total_steps}.'):
                            x = x.to(device)
                            y = y.to(device)
                            pred_depth = model(x)
                            valid_loss = criterion(pred_depth, y)
                            experiment.log_metric(f'valid_loss_epoch_{epoch + 1}', valid_loss, step=total_steps)

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

        * ✔️Normalize depth ==> divide by 2^16

        * add images to comet ml experiment.log_image(img, name=...)

        https://github.com/pytorch/vision/blob/e35793a1a4000db1f9f99673437c514e24e65451/torchvision/models/detection/roi_heads.py#L45
        box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    """
    epochs = 2
    batch_size = 1
    learning_rate = 0.0001
    downsample = 6

    model = UNet(downsample=downsample, in_channels=3, out_channels=1)
    # model.half()
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    """
    Use SGD and be the king of the word!
        * https://arxiv.org/pdf/1812.01187.pdf - practical - changes life
        * https://arxiv.org/pdf/1506.01186.pdf - some critique 
        * https://arxiv.org/pdf/1711.04623.pdf
        * http://proceedings.mlr.press/v70/arpit17a/arpit17a.pdf
        
    * Start with Adam with weight_decay (output has to be 0 - 1) + momentum + AMS grad
    * add LR scheduler
    """
    criterion = torch.nn.MSELoss()  # TODO: Consider SmoothL1
    device = 'cuda:7'

    dataset = create_dataset()
    test_size = int(0.1 * len(dataset))
    print(f'Creating test_dataset: {test_size} and train_dataset: {len(dataset) - test_size}')
    test_dataset, train_dataset = torch.utils.data.random_split(dataset, [test_size, len(dataset) - test_size])
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )

    # Log hyperparameters to Comet    
    experiment.log_parameters({
        downsample: downsample,
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
        test_dataloader=test_dataloader,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
