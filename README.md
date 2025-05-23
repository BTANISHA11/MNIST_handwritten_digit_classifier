# 🧠 Handwritten Digit Recognition with Neural Networks

This project involves building a neural network using **PyTorch** to recognize handwritten digits from the MNIST dataset. You'll preprocess the data, design a neural network, train and evaluate it, and finally save the trained model.

---

## 📦 Project Structure

- `notebook.ipynb`: Main Jupyter Notebook containing all code and explanations.
- `model.pth`: Saved trained model file (optional).
- `README.md`: Project documentation (this file).

---

## 🚀 Getting Started

Before beginning the project:
- Review the **project rubric**.
- Check the provided **Environment Setup Notebook**.
- Upload any additional files (like images or saved models) required for evaluation.

### 🛠 Prerequisites

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- Jupyter Notebook

Install dependencies with:
```bash
pip install torch torchvision matplotlib notebook
````


## 📋 Project Steps

### ✅ Step 1: Load and Preprocess the Dataset
- Load the MNIST dataset using `torchvision.datasets`.
- Convert images to tensors, normalize them, and flatten to 1D arrays.
- Use `DataLoader` for batching and shuffling the dataset.

### ✅ Step 2: Visualize and Explore the Data
- Visualize images either from the training `DataLoader` or with a separate `DataLoader` without preprocessing.
- Examine shape and dimension changes before and after transformation.
- Justify each preprocessing step to explain its necessity.

### ✅ Step 3: Build and Train Your Neural Network
- Define your model using `torch.nn.Module`.
- Use `CrossEntropyLoss` as the loss function.
- Choose an optimizer (e.g., `Adam`, `SGD`).
- Train your model using the training data.

### ✅ Step 4: Evaluate and Tune Your Model
- Evaluate the model’s performance on the test set.
- Tune hyperparameters such as:
  - Learning rate
  - Batch size
  - Number of epochs
  - Network architecture (layers, activations, etc.)
- Goal: **Achieve at least 90% accuracy** on the test dataset.

### ✅ Step 5: Save the Model
Save the trained model using:
```python
torch.save(model.state_dict(), 'model.pth')

