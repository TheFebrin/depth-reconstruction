from comet_ml import Experiment, ConfusionMatrix

import sys, os, inspect
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm import tqdm

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


def load_model():
    pass

def create_dataloader(batch_size: int) -> DataLoader:
    dataset = DepthDataset(
        color_images_path='../color/',
        depth_images_path='../depth/',
        transform=transforms.Compose([
            transforms.Resize((4 * 256, 7 * 256)),
            transforms.ToTensor(),
        ])
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )


def train(
    model,
    dataloader,
    epochs: int,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: str=None,
):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

    model.train()

    with experiment.train():
        print("[Comet ML] Log histogram of model weights. May take a few minutes.")
        # weights = []
        # for name in model.named_parameters():
        #     if 'weight' in name[0]:
        #         weights += name[1].detach().numpy().tolist()
        # experiment.log_histogram_3d(weights, step=0)

        total_steps: int = 0
        for epoch in range(epochs):
            # experiment.log_current_epoch(epoch)
            print(f'Epoch: {epoch + 1} / {epochs}')

            epoch_loss: float = 0.0
            steps: int = 0

            for x, y in dataloader:
                x = x.to(device)
                y = y.float().to(device)
        
                pred_depth = model(x)
                loss = criterion(pred_depth, y)
                epoch_loss += loss

                print('loss: ', loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                steps += 1
                total_steps += 1
                # experiment.log_metric("one_batch_loss", loss, step=total_steps)
                print('END')
                break
            
            # End of epoch
            print(f'Mean epoch {epoch + 1} loss: {epoch_loss / steps}')


def main() -> int:
    """
    TODO: 
        * Add argparse 
        * Data augmentation ?
            - normalize
            - flips 
            - rotations

        * Normalize output depth ??

        * add images to comet ml experiment.log_image(img, name=...)
    """
    epochs = 2
    batch_size = 16
    learning_rate = 0.0001
    downsample = 6

    model = UNet(downsample=downsample, in_channels=3, out_channels=1)
    dataloader = create_dataloader(batch_size=batch_size)
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    criterion = torch.nn.MSELoss()

    # Log hyperparameters to Comet    
    experiment.log_parameters({
        downsample: downsample,
        epochs: epochs,
        batch_size: batch_size,
        learning_rate: learning_rate,
    })
    experiment.set_model_graph(str(model))

    # for x, y in dataloader:
    #     print(x.shape)
    #     plt.imshow(np.dstack(x[0].numpy()))
    #     plt.show()
    #     break

    train(
        model=model, 
        dataloader=dataloader, 
        epochs=epochs, 
        criterion=criterion,
        optimizer=optimizer
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())