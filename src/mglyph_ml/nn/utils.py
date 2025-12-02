from torch import nn
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
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
    
    # Print summary statistics
    print("\nPrediction Summary:")
    print("=" * 50)
    
    with torch.no_grad():
        for i, sample_idx in enumerate(sample_indices):
            dataset_test.reset_transform()
            img_tensor, true_label = dataset_test[sample_idx]
            img_batch = img_tensor.unsqueeze(0).to(device)
            pred_label = model(img_batch).item()
            error = abs((pred_label - true_label) * 100)
            print(f"Sample {sample_idx}: True={true_label*100:.2f}, Pred={pred_label*100:.2f}, Error={error:.2f}")
    
    print(f"\nMean error on these {num_samples} samples: {np.mean(errors):.2f} x units")
    print(f"Max error: {np.max(errors):.2f} x units")
    print(f"Min error: {np.min(errors):.2f} x units")


def visualize_test_samples(
    importers: list[GlyphImporter],
    num_samples: int = 9,
    figsize: tuple[int, int] = (6, 6),
    augmentation_seed: int = 69
) -> None:
    """
    Visualize random samples from the test dataset (without predictions).
    
    Args:
        importers: List of GlyphImporter objects for the test set
        num_samples: Number of samples to visualize (default: 9 for 3x3 grid)
        figsize: Figure size for the plot (default: (6, 6))
        augmentation_seed: Random seed for reproducible augmentation (default: 69)
    
    Displays:
        - Grid of test images with their labels
    """
    # Create dataset without normalization for clear visualization
    temp_dataset = GlyphDataset(*importers, normalize=False, augmentation_seed=augmentation_seed)
    
    # Get random samples
    temp_dataset.reset_transform()
    sample_indices = random.sample(range(len(temp_dataset)), num_samples)
    
    # Calculate grid dimensions
    ncols = 3
    nrows = (num_samples + ncols - 1) // ncols  # Ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # Flatten for easy indexing
    
    for index, sample_idx in enumerate(sample_indices):
        image, label = temp_dataset[sample_idx]
        
        # Convert image for display
        img = image.numpy().clip(0, 1).transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        
        # Display the image
        axes[index].imshow(img)
        axes[index].set_title(f"{label:.2f}")
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()