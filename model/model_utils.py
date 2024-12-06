import numpy as np
import numpy.typing as npt

from typing import Tuple, List, Set


def bag_of_words_matrix(sentences: List[str]) -> npt.ArrayLike:
    """
    Convert the dataset into V x M matrix.
    M: number of examples
    V: vocabulary size
    """
    # Count word frequencies across all sentences
    word_freq = {}
    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Create vocabulary (including only words that appear >= 2 times)
    vocab = ['<UNK>'] + [word for word, freq in word_freq.items() if freq >= 2]
    vocab_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Create the V x M matrix
    V = len(vocab)
    M = len(sentences)
    bow_matrix = np.zeros((V, M))
    
    # Fill the matrix
    for col, sentence in enumerate(sentences):
        words = sentence.lower().split()
        for word in words:
            idx = vocab_to_idx.get(word, 0)
            bow_matrix[idx, col] += 1
            
    return bow_matrix


def labels_matrix(data: Tuple[List[str], Set[str]]) -> npt.ArrayLike:
    """
    Convert the dataset into K x M matrix.
    K: number of classes (unique intents)
    M: number of examples
    """
    labels, unique_labels = data
    
    # Create mapping from label to index
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    # Create K x M matrix
    K = len(unique_labels)
    M = len(labels)
    label_matrix = np.zeros((K, M))
    
    # Fill the matrix with one-hot encodings
    for col, label in enumerate(labels):
        label_matrix[label_to_idx[label], col] = 1
        
    return label_matrix


def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Softmax function.
    Formula: softmax(zi) = exp(zi) / sum(exp(zj)) for j=1 to K
    
    Args:
        z: Input array of shape (K, M) where K is number of classes and M is batch size
    
    Returns:
        Array of same shape as input with softmax probabilities
    """
    # Subtract maximum value for numerical stability
    shifted_z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.
    Formula: ReLU(z) = max(0,z)
    
    Args:
        z: Input array
    
    Returns:
        Array of same shape as input with ReLU activation applied
    """
    return np.maximum(0, z)


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.
    Formula: 
        1 if z > 0
        0 if z <= 0
    
    Args:
        z: Input array
    
    Returns:
        Array of same shape as input with ReLU derivative values
    """
    return (z > 0).astype(float)