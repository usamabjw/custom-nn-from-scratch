import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.params = {}
        self.init_weights()

    def init_weights(self):
        for i in range(1, len(self.layers)):
            self.params[f"W{i}"] = np.random.randn(self.layers[i], self.layers[i-1]) * 0.01
            self.params[f"b{i}"] = np.zeros((self.layers[i], 1))

    def forward(self, X):
        cache = {"A0": X}
        for i in range(1, len(self.layers)):
            Z = np.dot(self.params[f"W{i}"], cache[f"A{i-1}"]) + self.params[f"b{i}"]
            cache[f"A{i}"] = self.relu(Z) if i < len(self.layers) - 1 else self.softmax(Z)
        return cache[f"A{len(self.layers)-1}"], cache

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / exp_Z.sum(axis=0, keepdims=True)

    def compute_loss(self, Y, Y_hat):
        m = Y.shape[1]
        return -np.sum(Y * np.log(Y_hat + 1e-9)) / m

    def backward(self, X, Y, cache):
        grads = {}
        m = X.shape[1]
        A_prev = cache[f"A{len(self.layers)-2}"]
        dZ = cache[f"A{len(self.layers)-1}"] - Y

        for i in reversed(range(1, len(self.layers))):
            grads[f"dW{i}"] = np.dot(dZ, cache[f"A{i-1}"].T) / m
            grads[f"db{i}"] = np.sum(dZ, axis=1, keepdims=True) / m
            if i > 1:
                dA_prev = np.dot(self.params[f"W{i}"].T, dZ)
                dZ = dA_prev * (cache[f"A{i-1}"] > 0)

        return grads

    def update_params(self, grads, learning_rate):
        for i in range(1, len(self.layers)):
            self.params[f"W{i}"] -= learning_rate * grads[f"dW{i}"]
            self.params[f"b{i}"] -= learning_rate * grads[f"db{i}"]

    def fit(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            Y_hat, cache = self.forward(X)
            loss = self.compute_loss(Y, Y_hat)
            grads = self.backward(X, Y, cache)
            self.update_params(grads, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        Y_hat, _ = self.forward(X)
        return np.argmax(Y_hat, axis=0)
