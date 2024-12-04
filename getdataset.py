from sklearn.datasets import fetch_openml
import numpy as np

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)

# Convert the data and labels to numpy arrays
images = mnist.data.values.astype(np.float32)  # Convert to float32 to save memory
labels = mnist.target.values.astype(int)

# Calculate 5% of the dataset size
percent=0.05
fraction_size = int(percent * len(images))

# Select the first 20% of data
train_images, test_images = images[:fraction_size], images[fraction_size:]
train_labels, test_labels = labels[:fraction_size], labels[fraction_size:]

# Save the selected fraction as .npy files
np.save('data/train_images.npy', train_images)
np.save('data/train_labels.npy', train_labels)
np.save('data/test_images.npy', test_images)
np.save('data/test_labels.npy', test_labels)

print(f"Dataset fraction {percent*100}% saved as .npy files!")
