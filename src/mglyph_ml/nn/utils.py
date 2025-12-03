from torch import nn
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from mglyph_ml.data.glyph_dataset import GlyphDataset
from mglyph_ml.glyph_importer import GlyphImporter

def train_one_epoch(
    model: nn.Module,
    train_data_loader: DataLoader,
    device: str,
    criterion,  # unfortunately, has no supertype... There's torch.nn.modules._Loss, but it's private
    optimizer,  # same as above
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_error = 0.0
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
        error = torch.mean(torch.abs(outputs - labels)).item()

        running_loss += loss.item()
        running_error += error
        num_batches += 1
        # Optionally print every N batches
        # if (i + 1) % 10 == 0:
        # print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {loss.item():.4f}")

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_error = running_error / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_error


def evaluate_glyph_regressor(
    model: nn.Module, 
    data_loader: DataLoader, 
    device: str, 
    criterion
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
            inputs = inputs.to(device)
            labels = labels.to(device)
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


def visualize_test_predictions(
    model: nn.Module,
    importers: list[GlyphImporter],
    device: str,
    num_samples: int = 9,
    figsize: tuple[int, int] = (6, 6),
    augmentation_seed: int = 69
) -> None:
    """
    Visualize model predictions on random test samples.
    
    Args:
        model: The trained regression model to evaluate
        importers: List of GlyphImporter objects for the test set
        device: Device to run inference on ('cuda' or 'cpu')
        num_samples: Number of samples to visualize (default: 9 for 3x3 grid)
        figsize: Figure size for the plot (default: (6, 6))
        augmentation_seed: Random seed for reproducible augmentation (default: 69)
    
    Displays:
        - Grid of test images with true values, predictions, and errors
        - Summary statistics of the predictions
    """
    model.eval()
    model = model.to(device)
    
    # Create datasets - one normalized for prediction, one unnormalized for display
    dataset_test = GlyphDataset(*importers, augmentation_seed=augmentation_seed)
    temp_dataset_viz = GlyphDataset(*importers, normalize=False, augmentation_seed=augmentation_seed)
    
    # Get random samples from test set
    temp_dataset_viz.reset_transform()
    sample_indices = random.sample(range(len(temp_dataset_viz)), num_samples)
    
    # Calculate grid dimensions
    ncols = 3
    nrows = (num_samples + ncols - 1) // ncols  # Ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # Flatten for easy indexing
    
    errors = []
    
    with torch.no_grad():
        for index, sample_idx in enumerate(sample_indices):
            # Get sample from non-normalized dataset for display
            img_tensor_display, true_label = temp_dataset_viz[sample_idx]
            
            # Get same sample from normalized dataset for prediction
            dataset_test.reset_transform()
            img_tensor_pred, _ = dataset_test[sample_idx]
            
            # Make prediction
            img_batch = img_tensor_pred.unsqueeze(0).to(device)
            pred_label = model(img_batch).item()
            
            # Convert to x units (0-100)
            true_x = true_label * 100
            pred_x = pred_label * 100
            error = abs(pred_x - true_x)
            errors.append(error)
            
            # Convert image for display
            img = img_tensor_display.numpy().clip(0, 1).transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            
            # Display
            axes[index].imshow(img)
            axes[index].set_title(
                f'True: {true_x:.2f}\n'
                f'Pred: {pred_x:.2f}\n'
                f'Err: {error:.2f}',
                fontsize=9
            )
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def train_model_with_visualization(
    model: nn.Module,
    data_loader_train: DataLoader,
    data_loader_test: DataLoader,
    device: str,
    criterion,
    optimizer,
    num_epochs: int = 10,
    early_stopping_threshold: float = 0.3,
    figsize: tuple[int, int] = (6, 4),
    reset_test_transform: bool = True
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Train a model with real-time visualization of training progress.
    
    Args:
        model: The neural network model to train
        data_loader_train: DataLoader for training data
        data_loader_test: DataLoader for test/validation data
        device: Device to train on ('cuda' or 'cpu')
        criterion: Loss function
        optimizer: Optimizer for training
        num_epochs: Maximum number of epochs to train (default: 10)
        early_stopping_threshold: Stop training if test error drops below this value in x units (default: 0.3)
        figsize: Size of the visualization figure (default: (6, 4))
        reset_test_transform: Whether to reset test dataset transform before evaluation (default: True)
    
    Returns:
        Tuple of (train_losses, train_errors, test_losses, test_errors) - all as lists
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    
    losses = []
    errors = []
    test_losses = []
    test_errors = []
    
    # Train for specified number of epochs
    for epoch in range(num_epochs):
        loss, error = train_one_epoch(model, data_loader_train, device, criterion, optimizer)
        error *= 100.0  # Convert normalized error (0-1) to actual x units (0-100)
        losses.append(loss)
        errors.append(error)
        
        # Reset the dataloader's transform to make the test dataset 100% reproducible
        if reset_test_transform and hasattr(data_loader_train.dataset, 'reset_transform'):
            data_loader_train.dataset.reset_transform()
        if reset_test_transform and hasattr(data_loader_test.dataset, 'reset_transform'):
            data_loader_test.dataset.reset_transform()
        
        test_loss, test_error = evaluate_glyph_regressor(model, data_loader_test, device, criterion)
        test_error *= 100.0  # Convert normalized error (0-1) to actual x units (0-100)
        test_losses.append(test_loss)
        test_errors.append(test_error)
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        # Plot updated data with markers (dashed lines for loss, solid for error)
        ax1.plot(range(len(losses)), losses, color='green', label='Train Loss', marker='o', markersize=4, linestyle='--')
        ax2.plot(range(len(errors)), errors, color='green', label='Train Error', marker='o', markersize=4, linestyle='-')
        ax1.plot(range(len(test_losses)), test_losses, color='red', label='Test Loss', marker='o', markersize=4, linestyle='--')
        ax2.plot(range(len(test_errors)), test_errors, color='red', label='Test Error', marker='o', markersize=4, linestyle='-')
        
        ax1.grid(True, alpha=0.3)
        
        # Set labels and x-axis ticks
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)', color='black')
        ax2.set_ylabel('Error (Mean Absolute Error, x units)', color='black')
        ax2.yaxis.set_label_position('right')
        ax1.set_xticks(range(len(losses)))
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Update display
        clear_output(wait=True)
        
        # Early stopping: stop if error is good enough
        if test_error < early_stopping_threshold:
            print(f"Early stopping at epoch {epoch+1}: test error {test_error:.4f} x units is below threshold")
            display(fig)
            break
        
        display(fig)
    
    return losses, errors, test_losses, test_errors


def visualize_samples(
    plot_title: str,
    dataset: GlyphDataset,
    num_samples: int = 9,
    figsize: tuple[int, int] = (6, 6),
) -> plt.Figure:
    """
    Visualize random samples from the test dataset (without predictions).
    
    Args:
        plot_title: Title for the plot
        dataset: GlyphDataset to visualize samples from
        num_samples: Number of samples to visualize (default: 9 for 3x3 grid)
        figsize: Figure size for the plot (default: (6, 6))
    
    Returns:
        matplotlib.pyplot.Figure: The figure instance that can be displayed later
    
    Displays:
        - Grid of test images with their labels
    """
    # Get random samples
    dataset.reset_transform()
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    # Calculate grid dimensions
    ncols = 3
    nrows = (num_samples + ncols - 1) // ncols  # Ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # Flatten for easy indexing
    
    for index, sample_idx in enumerate(sample_indices):
        image, label = dataset[sample_idx]
        
        # Convert image for display
        img = image.numpy().clip(0, 1).transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        
        # Display the image
        axes[index].imshow(img)
        axes[index].set_title(f"{label:.2f}")
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(plot_title)
    fig.tight_layout()
    
    return fig