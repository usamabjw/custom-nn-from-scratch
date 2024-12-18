import numpy as np
from nn import NeuralNetwork
from utils import *

def load_data():
    train_images = np.load('data/train_images.npy').reshape(-1, 28*28).T / 255.0
    train_labels = np.load('data/train_labels.npy')
    test_images = np.load('data/test_images.npy').reshape(-1, 28*28).T / 255.0
    test_labels = np.load('data/test_labels.npy')
    
    train_labels_onehot = np.eye(10)[train_labels].T
    test_labels_onehot = np.eye(10)[test_labels].T
    return train_images, train_labels_onehot, test_images, test_labels_onehot

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data()
    nn = NeuralNetwork(layers=[784, 128, 64, 10])
    nn.fit(X_train, Y_train, epochs=100, learning_rate=0.5)
    Y_pred = nn.predict(X_test)
    print("Test Accuracy:", np.mean(Y_pred == np.argmax(Y_test, axis=0)))
    print("Prediction Visualization")
    Y_true = np.argmax(Y_test, axis=0)  # Convert one-hot encoded labels to class indices
    visualize_predictions(X_test, Y_true, Y_pred, num_samples=10, save_path="results/predictions.png")

