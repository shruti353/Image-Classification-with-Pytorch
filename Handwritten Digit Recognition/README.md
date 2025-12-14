# MNIST Digit Classification with CNN

## 📊 Results
- **Accuracy**: 98.65% on test set
- **Model**: CNN with 2 convolutional layers
- **Framework**: PyTorch

## 🚀 Quick Start
Run the Jupyter notebook `main.ipynb` to see:
- Data loading and preprocessing
- Model training
- Evaluation and visualization

## 📁 Files
- `main.ipynb` - Complete implementation
- `requirements.txt` - Dependencies

## 🧠 Model Architecture
Simple CNN with:
- Conv1: 10 filters, 5x5 kernel
- Conv2: 20 filters, 5x5 kernel
- Dropout: 0.5
- FC1: 320 → 50 neurons
- FC2: 50 → 10 neurons (output)

## 📈 Training
- Epochs: 10
- Batch size: 100
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss