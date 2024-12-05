import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_predictions(X, Y_true, Y_pred, num_samples=10, save_path=None):
    """
    Visualizes predictions by displaying sample images with predicted and true labels.
    Args:
        X (ndarray): Input data (images), shape (features, samples).
        Y_true (ndarray): True labels, shape (samples,).
        Y_pred (ndarray): Predicted labels, shape (samples,).
        num_samples (int): Number of samples to display.
        save_path (str): Path to save the visualization image (optional).
    """
    # Select random indices for visualization
    indices = np.random.choice(X.shape[1], size=num_samples, replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, idx in enumerate(indices):
        image = X[:, idx].reshape(28, 28)  # Reshape flat image to 28x28
        true_label = Y_true[idx]
        pred_label = Y_pred[idx]

        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to {save_path}")

    plt.show()
