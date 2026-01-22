# Fashion Image Classification using PyTorch

This project implements an image classification model using PyTorch to classify fashion products from the Fashion-MNIST dataset. The goal is to correctly identify the category of clothing items based on grayscale images.

---

## Dataset: Fashion-MNIST

Fashion-MNIST is a dataset of 28×28 grayscale images of clothing items, consisting of 60,000 training images and 10,000 test images across 10 classes.

### Classes
0 - T-shirt/top
1 - Trouser
2 - Pullover
3 - Dress
4 - Coat
5 - Sandal
6 - Shirt
7 - Sneaker
8 - Bag
9 - Ankle boot


---

## Project Overview

- Dataset loading using `torchvision.datasets`
- Data preprocessing and normalization
- Neural network model definition
- Model training using backpropagation
- Evaluation using test accuracy

---

## Model Architecture

- Input layer: 28 × 28 flattened images
- Fully connected hidden layers
- ReLU activation function
- Output layer with 10 classes
- Loss function: CrossEntropyLoss
- Optimizer: Adam

---

## Technology Stack

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Jupyter Notebook

---

## How to Run the Project

1. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib jupyter

2. Open the notebook:

jupyter notebook fashion_mnist_classification_nn_pytorch.ipynb


3. Run all cells to train and evaluate the model.

## Results

The model achieves good classification accuracy on the Fashion-MNIST test set

Demonstrates the effectiveness of neural networks for image classification tasks

Highlights the increased complexity of Fashion-MNIST compared to digit recognition

## Author

Shruti Thakkar

AI and Machine Learning Enthusiast

GitHub: https://github.com/shruti353
