# Image Classification with PyTorch

This repository contains two deep learning image classification projects implemented using PyTorch:

1. Handwritten Digit Recognition (MNIST)
2. Fashion Image Classification (Fashion-MNIST)

Both projects demonstrate the complete machine learning workflow, including data loading, model training, and evaluation.

---

## Project Structure

Image-Classification-with-Pytorch/
├── Handwritten Digit Recognition/
│ └── handwritten_digit_classification.ipynb
│
├── MNIST fashion classification/
│ └── fashion_mnist_classification_nn_pytorch.ipynb
│
└── README.md


---

## 1. Handwritten Digit Recognition (MNIST)

### Description
This project focuses on classifying handwritten digits from 0 to 9 using a neural network trained on the MNIST dataset.

### Key Features
- Implemented using PyTorch
- MNIST dataset with 28×28 grayscale images
- Neural network model training and evaluation
- Accuracy evaluation on test data

### Model Details
- Fully connected neural network
- Loss function: CrossEntropyLoss
- Optimizer: Adam or SGD

---

## 2. Fashion Image Classification (Fashion-MNIST)

### Description
This project classifies fashion products using the Fashion-MNIST dataset, which is more complex than digit recognition.

### Fashion Classes
0 → T-shirt/top
1 → Trouser
2 → Pullover
3 → Dress
4 → Coat
5 → Sandal
6 → Shirt
7 → Sneaker
8 → Bag
9 → Ankle boot


### Key Features
- PyTorch-based neural network
- Data preprocessing and normalization
- Training and validation accuracy evaluation

---

## Technology Stack

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Installation and Setup

1. Clone the repository:
git clone https://github.com/shruti353/Image-Classification-with-Pytorch.git
cd Image-Classification-with-Pytorch

2. Install dependencies:
pip install torch torchvision numpy matplotlib jupyter

3. Launch Jupyter Notebook:
jupyter notebook

## Results

Both models achieve strong classification performance

Fashion-MNIST is more challenging than MNIST digit classification

Demonstrates the effectiveness of neural networks for image data

## Author
Shruti Thakkar
AI and Machine Learning Enthusiast
GitHub: https://github.com/shruti353
