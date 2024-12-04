
# **Custom Neural Network from Scratch**

This project implements a fully functional neural network from scratch without using deep learning libraries like TensorFlow or PyTorch. The network is designed for a simple **image classification task** on the **MNIST dataset**.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Future Improvements](#future-improvements)
7. [Contributing](#contributing)
8. [License](#license)

---

## **Overview**

The goal of this project is to understand the inner workings of neural networks by implementing them from the ground up. The network supports:
- Multiple hidden layers with ReLU activations.
- A softmax output layer for classification.
- Backpropagation for learning.

This project showcases:
- Forward and backward propagation.
- Weight initialization, loss computation, and parameter updates.

---

## **Features**
- Implements neural network training **from scratch** using only NumPy.
- Includes support for:
  - ReLU and Softmax activation functions.
  - Multi-class cross-entropy loss.
  - Gradient descent optimization.
- Trains on the MNIST dataset (handwritten digit classification).

---

## **Installation**

### Prerequisites
Ensure you have Python 3.8+ installed. Install required packages using:

```bash
pip install -r requirements.txt
```

### Dataset
Download and prepare the MNIST dataset:
1. Convert MNIST into `.npy` format for easy loading (included in `data/`).
2. Place the files (`train_images.npy`, `train_labels.npy`, etc.) in the `data/` directory.

---

## **Usage**

### Run the Training Script
To train the neural network:
```bash
python src/train.py
```

### Configuration
Modify parameters such as learning rate, number of epochs, or architecture in `train.py`.

---

## **Results**

| Metric         | Value        |
|----------------|--------------|
| Training Accuracy | 98%         |
| Test Accuracy      | 97%         |

Sample predictions:
![Predictions Visualization](path_to_image.png)

---

## **Future Improvements**
- Add support for other activation functions (e.g., Sigmoid, Tanh).
- Implement additional optimizers like Adam or RMSProp.
- Extend to other datasets (e.g., CIFAR-10 or custom datasets).

---

## **Contributing**

Contributions are welcome! If you have ideas for improvement, please fork the repository and submit a pull request.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
