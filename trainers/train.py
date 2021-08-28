from comet_ml import Experiment, ConfusionMatrix
import torch
from tqdm import tqdm


def train(
    model,
    train_dataloader,
    epochs: int,
    criterion,
    optimizer: torch.optim.Optimizer,
    experiment: Experiment,
    calculate_valid_frequency: int,
    device: str=None,
):
    print('STARTING TRAINING')
    print(f'train_dataloader: {len(train_dataloader)}')

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

    with experiment.train():

        model = model.to(device)
        model.train()

        total_steps: int = 0
        best_epoch_loss: float = 1e18

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

                steps += 1
                total_steps += 1

            # End of epoch
            print(f'Mean epoch {epoch + 1} loss: {epoch_loss / steps}')

            if epoch_loss / steps < best_epoch_loss:
                best_epoch_loss = epoch_loss / steps
                print(f'Found better model. Loss: {best_epoch_loss} |  Epoch: {epoch} | total_steps: {total_steps}\n')
                torch.save(model.state_dict(), 'models/best_model.pth')

            experiment.log_metric(f'mean_loss_after_epoch', epoch_loss / steps, step=total_steps)