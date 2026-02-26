# Applied Machine Learning Experiments

This repository contains selected deep learning and representation learning projects focused on model implementation, fine-tuning, and benchmarking. The work emphasizes hands-on experimentation, optimization, and evaluation of modern machine learning systems using PyTorch and NumPy.

## Installation

Clone the repository and install the required dependencies:

```
git clone https://github.com/duruaran/applied-ml-experiments.git
cd applied-ml-experiments
pip install -r requirements.txt
```

## Projects
### Transformer-Based Text Classification and Fine-Tuning (BERT)

Implemented and evaluated a transformer-based text classification system using BERT (bert-base-uncased) on the AG News dataset.

### Data
Datasets are not included in this repository.

* AG News is automatically downloaded using the HuggingFace `datasets` library.
* KMNIST should be placed in the `data/` directory if running locally.

Instructions for downloading datasets are included in the notebooks.

#### Key Components
* Extracted frozen BERT embeddings using multiple probing strategies (CLS token, mean pooling, etc.)
* Benchmarked KNN and multi-class logistic regression on learned representations
* Fine-tuned all BERT parameters end-to-end using PyTorch
* Compared probing versus full fine-tuning performance
* Visualized attention weights to analyze model behavior

#### Focus Areas
* Representation learning
* Transfer learning
* Transformer fine-tuning
* Attention analysis
* Model benchmarking

### Neural Network Implementation and Vision Benchmarking (KMNIST)

Built and evaluated neural network architectures for image classification on the Kuzushiji-MNIST dataset.

#### From-Scratch MLP Implementation
* Implemented a multilayer perceptron in NumPy
* Backpropagation and mini-batch stochastic gradient descent
* Gradient checking for correctness verification
* L2 regularization
* Hyperparameter experimentation (depth, width, activation functions)

#### CNN Benchmarking (PyTorch)
* Designed convolutional neural networks
* Compared MLP and CNN performance
* Evaluated the impact of depth, activation choice, and regularization

#### Focus Areas
* Optimization and gradient-based learning
* Architecture comparison (MLP vs CNN)
* Experimental design and evaluation
* Model performance analysis

## Tech Stack
* Python
* PyTorch
* NumPy
* scikit-learn
* HuggingFace Transformers

## Goals
These projects explore:
* The impact of pretrained representations on downstream performance
* Architectural depth and non-linearity in neural networks
* Trade-offs between frozen embeddings and end-to-end fine-tuning
* Practical implementation of neural networks from first principles