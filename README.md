# Machine Learning Projects

This repository contains three distinct AI model implementations without external libraries:

1. **Neural Network** 
2. **Polynomial Regression with Regularization** 
3. **Naive Bayes Spam Classifier**
---
##  Neural Network (Binary Classification)

A feedforward neural network built from scratch with backpropagation and sigmoid activations. Used for classifying 2D points into binary labels.
### Data format
```
x1,x2,Label
.
.
.
```
### Usage

```bash
python NeuralNetwork.py
```

## Polynomial Regression with L2 Regularization

A batch-based gradient descent regressor for fitting polynomial curves to data, with experiments on model complexity and regularization.

### Data format
```
{
  "X_train": [n1,n2, ... ],
  "Y_train": [n1,n2,...],
  "X_val": [n1,n2,...],
  "Y_val": [n1,n2,...],
  "X_test": [n1,n2,...],
  "Y_test": [n1,n2,...]
}
```
### Usage
```
python Regression.py
```
## Naive Bayes Spam Classifier

A custom Naive Bayes classifier for text classification of email files into spam or ham categories.

### Data format
```
├── ham/
│   └── 1.text
│   └── 2.text
│   └── 3.text
│   └── .
│   └── .
│   └── .
├── spam/
│   └── 1.text
│   └── 2.text
│   └── 3.text
│   └── .
│   └── .
│   └── .
└── test/
│   └── 1.text
│   └── 2.text
│   └── 3.text
│   └── .
│   └── .
│   └── .
```
### Usage
```
python BayesianModel.py
```
