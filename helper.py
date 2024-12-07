
from model.ffnn import NeuralNetwork, compute_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def _calculate_accuracy(model, X, Y):
    """Helper function to calculate prediction accuracy"""
    predictions = model.predict(X)
    pred_classes = np.argmax(predictions, axis=0)
    true_classes = np.argmax(Y, axis=0)
    return np.mean(pred_classes == true_classes)

def _update_model_parameters(model, learning_rate, gradients):
    """Helper function to update model parameters"""
    dW1, db1, dW2, db2 = gradients
    model.W1 -= learning_rate * dW1
    model.b1 -= learning_rate * db1
    model.W2 -= learning_rate * dW2
    model.b2 -= learning_rate * db2
    

def batch_train(X, Y, model, train_flag=False):
    ################################# STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) Use your neural network to predict the intent
    #         (without any training) and calculate the accuracy 
    #         of the classifier. Should you be expecting high
    #         numbers yet?
    #         2) if train_flag is true, run the training for 1000 epochs using 
    #         learning rate = 0.005 and use this neural network to predict the 
    #         intent and calculate the accuracy of the classifier
    #         3) Then, plot the cost function for each iteration and
    #         compare the results after training with results before training

    # Calculate initial accuracy before training
    initial_accuracy = _calculate_accuracy(model, X, Y)
    print(f"Initial accuracy (before training): {initial_accuracy * 100:.2f}%")

    if not train_flag:
        return initial_accuracy

    learning_rate = 0.005
    epochs = 1000
    costs = []

    # Train the model
    for epoch in range(epochs):
        predictions = model.predict(X) # Forward pass
        cost = compute_loss(predictions, Y) # calculate loss
        costs.append(cost)
        
        gradients = model.backward(X, Y) # Backward pass
        _update_model_parameters(model, learning_rate, gradients)

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")

    # Calculate final accuracy after training
    final_accuracy = _calculate_accuracy(model, X, Y)
    
    # Plot the cost vs. iteration
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title('Cost vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.savefig('training_cost.png')
    plt.close()
    
    print("\nResults Comparison:")
    print(f"Before training accuracy: {initial_accuracy * 100:.2f}%")
    print(f"After training accuracy: {final_accuracy * 100:.2f}%")
    
    return final_accuracy
    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False):
    """
    Train the neural network using mini-batch and SGD approaches.
    
    Args:
        X: Input matrix of shape (input_size, M)
        Y: Ground truth matrix of shape (num_classes, M)
        model: Neural network model instance
        train_flag: Whether to train the model
    
    Returns:
        Initial accuracy if not training, final accuracy if training
    """
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.
    
    initial_accuracy = _calculate_accuracy(model, X, Y)
    print(f"Initial accuracy: {initial_accuracy * 100:.2f}%")

    if not train_flag:
        return initial_accuracy

    learning_rate = 0.005
    epochs = 1000
    batch_sizes = [64, 1] # Mini-batch and SGD
    M = X.shape[1] # Number of training examples
    costs_dict = {f'batch_size_{size}': [] for size in batch_sizes}
    final_accuracies = {}

    # Train the model with both batch sizes
    for batch_size in batch_sizes:
        print(f"\nTraining with batch size: {batch_size}")
        # Initialize new model for each training
        model = NeuralNetwork(model.W1.shape[1], model.W1.shape[0], model.W2.shape[0])

        # Train the model
        for epoch in range(epochs):
            epoch_cost = _train_one_epoch(X, Y, model, M, batch_size, learning_rate)
            costs_dict[f'batch_size_{batch_size}'].append(epoch_cost)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {epoch_cost:.4f}")

        final_accuracy = _calculate_accuracy(model, X, Y)
        final_accuracies[batch_size] = final_accuracy
        print(f"Final accuracy (batch_size={batch_size}): {final_accuracy * 100:.2f}%")

    # Plot the cost vs. iteration for both batch sizes
    plt.figure(figsize=(12, 6))
    for batch_size in batch_sizes:
        costs = costs_dict[f'batch_size_{batch_size}']
        label = 'Mini-batch (size=64)' if batch_size == 64 else 'SGD (size=1)'
        plt.plot(costs, label=label)
    
    plt.title('Cost vs. Iteration for Different Batch Sizes')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.savefig('minibatch_comparison.png')
    plt.close()

    return final_accuracies
    #########################################################################

def _train_one_epoch(X, Y, model, M, batch_size, learning_rate):
    """Helper function to train one epoch"""
    epoch_cost = 0
    indices = np.random.permutation(M) # Shuffle indices for mini-batch
    
    # Iterate over mini-batches
    for start_idx in range(0, M, batch_size):
        end_idx = min(start_idx + batch_size, M)
        batch_indices = indices[start_idx:end_idx]
        
        # Get currrent batch
        X_batch = X[:, batch_indices]
        Y_batch = Y[:, batch_indices]
        
        # Forward and backward pass
        predictions = model.predict(X_batch)
        cost = compute_loss(predictions, Y_batch)
        epoch_cost += cost * (end_idx - start_idx)
        
        # Update model parameters
        gradients = model.backward(X_batch, Y_batch)
        _update_model_parameters(model, learning_rate, gradients)
    
    return epoch_cost / M # Return average cost for the epoch

    