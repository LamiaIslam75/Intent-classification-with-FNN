
from model.ffnn import NeuralNetwork, compute_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


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

    # 1) Initial prediction and accuracy (before training)
    predictions = model.predict(X)
    pred_classes = np.argmax(predictions, axis=0)
    true_classes = np.argmax(Y, axis=0)
    initial_accuracy = np.mean(pred_classes == true_classes)
    print(f"Initial accuracy (before training): {initial_accuracy * 100:.2f}%")

    if train_flag:
        # 2) Training loop
        learning_rate = 0.005
        epochs = 1000
        costs = []  # To store cost values for plotting
        
        for epoch in range(epochs):
            predictions = model.predict(X)
            
            # Compute cost
            cost = compute_loss(predictions, Y)
            costs.append(cost)
            
            # Backward pass and update weights
            dW1, db1, dW2, db2 = model.backward(X, Y)
            
            # Update parameters
            model.W1 -= learning_rate * dW1
            model.b1 -= learning_rate * db1
            model.W2 -= learning_rate * dW2
            model.b2 -= learning_rate * db2
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")

        # Evaluate after training
        final_predictions = model.predict(X)
        final_pred_classes = np.argmax(final_predictions, axis=0)
        final_accuracy = np.mean(final_pred_classes == true_classes)
        print(f"Final accuracy (after training): {final_accuracy * 100:.2f}%")
        
        # 3) Plot the cost function
        plt.figure(figsize=(10, 6))
        plt.plot(costs)
        plt.title('Cost vs. Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.savefig('training_cost.png')
        plt.close()
        
        print("\nCost plot has been saved as 'training_cost.png'")
        
        # Compare results
        print("\nResults Comparison:")
        print(f"Before training accuracy: {initial_accuracy * 100:.2f}%")
        print(f"After training accuracy: {final_accuracy * 100:.2f}%")
        
    return initial_accuracy if not train_flag else final_accuracy
    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False):
    """
    Train the neural network using mini-batch and SGD approaches.
    
    Args:
        X: Input matrix of shape (input_size, M)
        Y: Ground truth matrix of shape (num_classes, M)
        model: Neural network model instance
        train_flag: Whether to train the model
    """
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.
    
    if not train_flag:
        return
        
    learning_rate = 0.005
    epochs = 1000
    batch_sizes = [64, 1]  # mini-batch and SGD
    M = X.shape[1]  # number of examples
    
    # Store costs for both approaches
    costs_dict = {f'batch_size_{size}': [] for size in batch_sizes}
    
    # Train with different batch sizes
    for batch_size in batch_sizes:
        print(f"\nTraining with batch size: {batch_size}")
        
        # Reset model for fair comparison
        model.__init__(model.W1.shape[1], model.W1.shape[0], model.W2.shape[0])
        
        # Training loop
        for epoch in range(epochs):
            epoch_cost = 0
            # Generate random indices for shuffling
            indices = np.random.permutation(M)
            
            # Mini-batch training
            for start_idx in range(0, M, batch_size):
                end_idx = min(start_idx + batch_size, M)
                batch_indices = indices[start_idx:end_idx]
                
                # Get current batch
                X_batch = X[:, batch_indices]
                Y_batch = Y[:, batch_indices]
                
                predictions = model.predict(X_batch)
                
                # Compute cost
                cost = compute_loss(predictions, Y_batch)
                epoch_cost += cost * (end_idx - start_idx)
                
                # Backward pass and update weights
                dW1, db1, dW2, db2 = model.backward(X_batch, Y_batch)
                
                # Update parameters
                model.W1 -= learning_rate * dW1
                model.b1 -= learning_rate * db1
                model.W2 -= learning_rate * dW2
                model.b2 -= learning_rate * db2
            
            # Average epoch cost
            epoch_cost /= M
            costs_dict[f'batch_size_{batch_size}'].append(epoch_cost)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {epoch_cost:.4f}")
    
    # Plot costs for both approaches
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
    
    print("\nTraining plot has been saved as 'minibatch_comparison.png'")
    #########################################################################
