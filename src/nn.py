import numpy as np

class NeuralNetwork:
    def __init__(self, layers, seed=None):
        """
        Initializes the neural network.
        Args:
            layers (list): List containing the number of neurons in each layer.
            seed (int): Random seed for reproducibility.
        """
        self.layers = layers
        self.params = {}
        if seed:
            np.random.seed(seed)
        self._init_weights()

    def _init_weights(self):
        """Initializes weights and biases for each layer."""
        for i in range(1, len(self.layers)):
            self.params[f"W{i}"] = np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2 / self.layers[i-1])
            self.params[f"b{i}"] = np.zeros((self.layers[i], 1))

    def forward(self, X):
        """
        Forward propagation through the network.
        Args:
            X (ndarray): Input data of shape (features, samples).
        Returns:
            A_last (ndarray): Output probabilities from the last layer.
            cache (dict): Intermediate values for backpropagation.
        """
        cache = {"A0": X}
        for i in range(1, len(self.layers)):
            Z = np.dot(self.params[f"W{i}"], cache[f"A{i-1}"]) + self.params[f"b{i}"]
            cache[f"A{i}"] = self._relu(Z) if i < len(self.layers) - 1 else self._softmax(Z)
        return cache[f"A{len(self.layers)-1}"], cache

    def _relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)

    def _softmax(self, Z):
        """Softmax activation function."""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / exp_Z.sum(axis=0, keepdims=True)

    def compute_loss(self, Y, Y_hat):
        """
        Computes the categorical cross-entropy loss.
        Args:
            Y (ndarray): One-hot encoded true labels of shape (classes, samples).
            Y_hat (ndarray): Predicted probabilities of shape (classes, samples).
        Returns:
            float: Loss value.
        """
        m = Y.shape[1]
        return -np.sum(Y * np.log(Y_hat + 1e-9)) / m

    def backward(self, Y, cache):
        """
        Backward propagation through the network.
        Args:
            Y (ndarray): One-hot encoded true labels.
            cache (dict): Intermediate values from forward propagation.
        Returns:
            grads (dict): Gradients of weights and biases for each layer.
        """
        grads = {}
        m = Y.shape[1]
        dZ = cache[f"A{len(self.layers)-1}"] - Y

        for i in reversed(range(1, len(self.layers))):
            grads[f"dW{i}"] = np.dot(dZ, cache[f"A{i-1}"].T) / m
            grads[f"db{i}"] = np.sum(dZ, axis=1, keepdims=True) / m
            if i > 1:
                dA_prev = np.dot(self.params[f"W{i}"].T, dZ)
                dZ = dA_prev * (cache[f"A{i-1}"] > 0)

        return grads

    def update_params(self, grads, learning_rate):
        """
        Updates the weights and biases using gradient descent.
        Args:
            grads (dict): Gradients of weights and biases.
            learning_rate (float): Learning rate for parameter updates.
        """
        for i in range(1, len(self.layers)):
            self.params[f"W{i}"] -= learning_rate * grads[f"dW{i}"]
            self.params[f"b{i}"] -= learning_rate * grads[f"db{i}"]

    def fit(self, X, Y, epochs, learning_rate, verbose=True):
        """
        Trains the neural network.
        Args:
            X (ndarray): Training data.
            Y (ndarray): One-hot encoded true labels.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for parameter updates.
            verbose (bool): Whether to print loss at intervals.
        """
        for epoch in range(epochs):
            Y_hat, cache = self.forward(X)
            loss = self.compute_loss(Y, Y_hat)
            grads = self.backward(Y, cache)
            self.update_params(grads, learning_rate)
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Predicts class labels for input data.
        Args:
            X (ndarray): Input data.
        Returns:
            ndarray: Predicted class labels.
        """
        Y_hat, _ = self.forward(X)
        return np.argmax(Y_hat, axis=0)

    def accuracy(self, Y_true, Y_pred):
        """
        Calculates the accuracy of predictions.
        Args:
            Y_true (ndarray): True labels.
            Y_pred (ndarray): Predicted labels.
        Returns:
            float: Accuracy value.
        """
        if Y_true.size == 0 or Y_pred.size == 0:
            print("Error: One or both input arrays are empty.")
            return 0.0
        if Y_true.shape != Y_pred.shape:
            print(f"Error: Shape mismatch - Y_true: {Y_true.shape}, Y_pred: {Y_pred.shape}")
            return 0.0
        return np.mean(Y_true == Y_pred)
