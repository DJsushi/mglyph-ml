import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate_glyph_regressor(
    model: nn.Module, data_loader: DataLoader, device: str, criterion
) -> tuple[float, float]:
    """
    Takes a glyph regressor, temporarily disables gradient calculation, and calculates the average
    loss on the given dataset (DataLoader). Processes in batches on GPU for efficiency.

    Returns a tuple containing the average loss and average error (mean absolute error)
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    running_error = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels = labels.view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate error as the average absolute difference (y_hat - y)
            error = torch.mean(torch.abs(outputs - labels)).item()

            running_loss += loss.item()
            running_error += error
            num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_error = running_error / num_batches if num_batches > 0 else 0.0
    model.train()  # Set model back to training mode
    return avg_loss, avg_error
