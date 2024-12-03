# Intent Classification with Feedforward Neural Network

[Overview](#overview) | [Features](#features) | [Requirements](#requirements) | [Project Structure](#project-structure) | [Usage](#usage) | [Implementation Details](#implementation-details) | [Dataset](#dataset)

## Overview
Implementation of a Feedforward Neural Network for intent classification using only Python and NumPy. The model classifies user intents from text input using the Sonos NLU Benchmark dataset.

## Features
- Bag-of-Words text representation
- Neural Network implementation with:
  - Input layer
  - Hidden layer with ReLU activation
  - Output layer with Softmax activation
- Training methods:
  - Batch gradient descent
  - Mini-batch gradient descent
  - Stochastic gradient descent

## Requirements
- Python
- NumPy
- matplotlib (for plotting)

Install requirements:
```bash
pip install -r requirements.txt
```

## Project Structure
```
assignment2/
├── data/
│   └── dataset.csv
├── model/
│   ├── __init__.py
│   ├── ffnn.py
│   └── model_utils.py
├── assignment2.py
├── utils.py
└── helper.py
```

## Usage
1. Basic training with batch gradient descent:
```bash
python assignment2.py --train
```

2. Training with mini-batch/stochastic gradient descent:
```bash
python assignment2.py --minibatch --train
```

## Implementation Details
- Learning rate: 0.005
- Number of epochs: 1000
- Mini-batch size: 64 (for mini-batch mode)
- Hidden layer size: 150 neurons
- Activation functions: ReLU (hidden layer), Softmax (output layer)
- Loss function: Cross-entropy

## Dataset
Uses the Sonos NLU Benchmark dataset under Creative Commons Zero v1.0 Universal license.
