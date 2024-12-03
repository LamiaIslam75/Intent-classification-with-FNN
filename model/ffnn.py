import numpy as np
import numpy.typing as npt

from model.model_utils import softmax, relu, relu_prime
from typing import Tuple


class NeuralNetwork(object):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int, 
        num_classes: int,
        seed: int = 1
    ):
        """
        Initialize neural network's weights and biases.
        Args:
            input_size: dimension of input features (V)
            hidden_size: number of neurons in hidden layer (150)
            num_classes: number of output classes (K)
            seed: random seed for reproducibility
        """
        ############################# STUDENT SOLUTION ####################
        # YOUR CODE HERE
        #     TODO:
        #         1) Set a seed so that your model is reproducible
        #         2) Initialize weight matrices and biases with uniform
        #         distribution in the range (-1, 1).

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Initialize weights and biases for hidden layer
        # W1 shape: (hidden_size, input_size)
        # b1 shape: (hidden_size, 1)
        self.W1 = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.b1 = np.random.uniform(-1, 1, (hidden_size, 1))
        
        # Initialize weights and biases for output layer
        # W2 shape: (num_classes, hidden_size)
        # b2 shape: (num_classes, 1)
        self.W2 = np.random.uniform(-1, 1, (num_classes, hidden_size))
        self.b2 = np.random.uniform(-1, 1, (num_classes, 1))
        ###################################################################

    def forward(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Forward pass with X as input matrix, returning the model prediction
        Y_hat.
        Args:
            X: Input matrix of shape (input_size, M) where M is batch size
        
        Returns:
            Y_hat: Output predictions of shape (num_classes, M)
        """
        ######################### STUDENT SOLUTION #########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform only a forward pass with X as input.
        
        # First layer forward pass
        # z1 = W1 @ X + b1
        self.z1 = np.dot(self.W1, X) + self.b1  # shape: (hidden_size, M)
        self.a1 = relu(self.z1)  # shape: (hidden_size, M)
        
        # Second layer forward pass
        # z2 = W2 @ a1 + b2
        self.z2 = np.dot(self.W2, self.a1) + self.b2  # shape: (num_classes, M)
        self.a2 = softmax(self.z2)  # shape: (num_classes, M)

        return self.a2
        #####################################################################

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Create a prediction matrix with `self.forward()`
        """
        ######################### STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Create a prediction matrix of the intent data using
        #         `self.forward()` function. The shape of prediction matrix
        #         should be similar to label matrix produced with
        #         `labels_matrix()`

        return self.forward(X)
        ######################################################################

    def backward(
        self, 
        X: npt.ArrayLike, 
        Y: npt.ArrayLike
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Backpropagation algorithm.

        Args:
            X: Input matrix (input_size x M)
            Y: Ground truth matrix (num_classes x M)
        
        Returns:
            Tuple of gradients (dW1, db1, dW2, db2)
        """
        ########################## STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform forward pass, then backpropagation
        #         to get gradient for weight matrices and biases
        #         2) Return the gradient for weight matrices and biases
        
        # Get batch size
        M = X.shape[1]
        
        # Forward pass (ensure we have the latest activations)
        self.forward(X)
        
        # Output layer error (δL = ŷ - y)
        delta2 = self.a2 - Y  # shape: (num_classes x M)
        
        # Hidden layer error (δl = (Wl+1)Tδl+1 ⊙ g'(zl))
        delta1 = np.dot(self.W2.T, delta2) * relu_prime(self.z1)  # shape: (hidden_size x M)
        
        # Compute gradients
        # ∂L/∂W2 = δ2(a1)T
        dW2 = np.dot(delta2, self.a1.T) / M
        # ∂L/∂b2 = δ2
        db2 = np.sum(delta2, axis=1, keepdims=True) / M
        
        # ∂L/∂W1 = δ1(X)T
        dW1 = np.dot(delta1, X.T) / M
        # ∂L/∂b1 = δ1
        db1 = np.sum(delta1, axis=1, keepdims=True) / M
        
        return dW1, db1, dW2, db2
        #######################################################################


def compute_loss(pred: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """
    Compute the cross entropy loss.
    Args:
        pred: Predicted probabilities (K x M)
        truth: One-hot encoded ground truth (K x M)
    
    Returns:
        Average cross entropy loss
    """
    ########################## STUDENT SOLUTION ###########################
    # YOUR CODE HERE
    #     TODO:
    #         1) Compute the cross entropy loss between your model prediction
    #         and the ground truth.

    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1 - epsilon)
    
    # Compute cross entropy loss
    M = truth.shape[1]  # number of examples
    cross_entropy = -np.sum(truth * np.log(pred)) / M
    
    return cross_entropy
    #######################################################################