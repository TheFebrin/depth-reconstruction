from comet_ml import Experiment, ConfusionMatrix
import torch
from tqdm import tqdm


def train(
    model,
    train_dataloader,
    test_synthetic_dataloader,
    test_production_dataloader,
    epochs: int,
    criterion,
    optimizer: torch.optim.Optimizer,
    experiment: Experiment,
    calculate_valid_frequency: int,
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
        best_synthetic_loss: float = 1e18
        best_production_loss: float = 1e18

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

                # print(f'\n\n[STEPS]: {total_steps} | Loss: {loss}')
                # print('Shape and dtype: ', pred.shape, pred.dtype)
                # print('Mean and std: ', pred.mean(), pred.std())
                # print(pred)
                # plt.imshow(pred[0].detach().cpu().permute(1, 2, 0))
                # plt.show()
                #
                # print('True Y:', y[0].shape)
                # print(y)
                # plt.imshow(y[0].detach().cpu().permute(1, 2, 0).cpu())
                # plt.show()

                experiment.log_metric("one_batch_loss", loss, step=total_steps)
                epoch_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (total_steps + 1) % calculate_valid_frequency == 0:
                    model.eval()
                    with torch.no_grad():
                        # save images from training set
                        experiment.log_image(x[0].detach().cpu().permute(1, 2, 0), name=f'train_color_image_{total_steps}')
                        experiment.log_image(y[0].detach().cpu().permute(1, 2, 0), name=f'train_normals_image_{total_steps}')
                        experiment.log_image(pred[0].detach().cpu().permute(1, 2, 0), name=f'train_prediction_image_{total_steps}')

                        # save mean loss
                        experiment.log_metric(f'mean_loss', epoch_loss / steps, step=total_steps)

                        valid_loss: float = 0.0
                        for x, y in tqdm(test_synthetic_dataloader, desc=f'Validating on synthetic test set. Epoch: {epoch + 1} | Total steps: {total_steps}.'):
                            x = x.to(device)
                            y = y.to(device)
                            pred = model(x)
                            loss = criterion(pred, y)
                            valid_loss += loss

                        valid_loss /= len(test_synthetic_dataloader)
                        if valid_loss < best_synthetic_loss:
                            best_synthetic_loss = valid_loss
                            torch.save(model.state_dict(), 'models/best_synthetic_model.pth')

                        print(f'Epoch: {epoch} | Step: {total_steps} | Synthetic loss: {valid_loss}')
                        experiment.log_metric(f'valid_synthetic_loss', valid_loss, step=total_steps)

                        valid_loss = 0.0
                        for x, y in tqdm(test_production_dataloader, desc=f'Validating on production test set. Epoch: {epoch + 1} | Total steps: {total_steps}.'):
                            x = x.to(device)
                            y = y.to(device)
                            pred = model(x)
                            loss = criterion(pred, y)
                            valid_loss += loss

                        valid_loss /= len(test_production_dataloader)
                        if valid_loss < best_production_loss:
                            best_production_loss = valid_loss
                            torch.save(model.state_dict(), 'models/best_prodiction_model.pth')
                        print(f'Epoch: {epoch} | Step: {total_steps} | Production loss: {valid_loss}')
                        experiment.log_metric(f'valid_production_loss', valid_loss, step=total_steps)

                    model.train()

                steps += 1
                total_steps += 1

            # End of epoch
            print(f'Mean epoch {epoch + 1} loss: {epoch_loss / steps}')
            experiment.log_metric(f'mean_loss_after_epoch', epoch_loss / steps, step=total_steps)