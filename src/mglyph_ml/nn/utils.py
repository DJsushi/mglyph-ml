from torch import nn
from torch.utils.data import DataLoader
import torch

def train_one_epoch(
    model: nn.Module,
    train_data_loader: DataLoader,
    device: str,
    criterion,  # unfortunately, has no supertype... There's torch.nn.modules._Loss, but it's private
    optimizer,  # same as above
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = 0

    for index, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy as the average absolute difference (y_hat - y)
        accuracy = torch.mean(torch.abs(outputs - labels)).item()

        running_loss += loss.item()
        running_accuracy += accuracy
        num_batches += 1
        # Optionally print every N batches
        # if (i + 1) % 10 == 0:
        # print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {loss.item():.4f}")

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_accuracy = running_accuracy / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_accuracy