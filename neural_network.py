import numpy as np
import pickle

# Constants and Hyperparameters
INPUT_SIZE = 10
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
HIDDEN_SIZE_3 = 32
OUTPUT_SIZE = 10
EPOCHS = 10000
INITIAL_LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.00001
LEARNING_RATE_DECAY = 0.99
DROPOUT_RATE = 0.2
GRADIENT_CLIP_THRESHOLD = 5.0
EARLY_STOPPING_PATIENCE = 500
MODEL_FILENAME = "trained_model.pkl"

# Activation Functions and Their Derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def elu(x):
    return np.where(x > 0, x, np.exp(x) - 1)

def elu_derivative(x):
    return np.where(x > 0, 1, elu(x) + 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + swish(x) * (1 - s)

# Softmax and Cross-Entropy Loss
def softmax(output):
    max_val = np.max(output)
    exp_vals = np.exp(output - max_val)
    return exp_vals / np.sum(exp_vals)

def cross_entropy_loss(predicted, actual):
    return -np.sum(actual * np.log(predicted + 1e-15))

# Weight Initialization
def initialize_weights(input_size, output_size):
    return np.random.uniform(-1, 1, input_size * output_size).reshape(input_size, output_size) / np.sqrt(input_size)

# Forward Pass
def forward_pass(input, weights_hidden1, weights_hidden2, weights_hidden3, weights_output,
                  biases_hidden1, biases_hidden2, biases_hidden3, biases_output):
    hidden1_output = np.dot(input, weights_hidden1) + biases_hidden1
    hidden1_output = swish(hidden1_output)
    
    hidden2_output = np.dot(hidden1_output, weights_hidden2) + biases_hidden2
    hidden2_output = relu(hidden2_output)
    
    hidden3_output = np.dot(hidden2_output, weights_hidden3) + biases_hidden3
    hidden3_output = elu(hidden3_output)
    
    final_output = np.dot(hidden3_output, weights_output) + biases_output
    final_output = softmax(final_output)
    
    return hidden1_output, hidden2_output, hidden3_output, final_output

# Backpropagation
def backpropagate(input, hidden1_output, hidden2_output, hidden3_output, final_output, actual_output,
                   weights_hidden1, weights_hidden2, weights_hidden3, weights_output,
                   biases_hidden1, biases_hidden2, biases_hidden3, biases_output, learning_rate):
    
    output_gradient = final_output - actual_output
    
    # Update weights and biases for Output Layer
    weights_output -= learning_rate * np.outer(hidden3_output, output_gradient)
    biases_output -= learning_rate * output_gradient

    # Hidden Layer 3 Gradients
    hidden3_gradient = np.dot(weights_output, output_gradient) * elu_derivative(hidden3_output)
    
    # Update weights and biases for Hidden Layer 3
    weights_hidden3 -= learning_rate * np.outer(hidden2_output, hidden3_gradient)
    biases_hidden3 -= learning_rate * hidden3_gradient
    
    # Hidden Layer 2 Gradients
    hidden2_gradient = np.dot(weights_hidden3, hidden3_gradient) * relu_derivative(hidden2_output)
    
    # Update weights and biases for Hidden Layer 2
    weights_hidden2 -= learning_rate * np.outer(hidden1_output, hidden2_gradient)
    biases_hidden2 -= learning_rate * hidden2_gradient
    
    # Hidden Layer 1 Gradients
    hidden1_gradient = np.dot(weights_hidden2, hidden2_gradient) * swish_derivative(hidden1_output)
    
    # Update weights and biases for Hidden Layer 1
    weights_hidden1 -= learning_rate * np.outer(input, hidden1_gradient)
    biases_hidden1 -= learning_rate * hidden1_gradient

# Training Process
def train(inputs, targets, num_samples):
    weights_hidden1 = initialize_weights(INPUT_SIZE, HIDDEN_SIZE_1)
    weights_hidden2 = initialize_weights(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
    weights_hidden3 = initialize_weights(HIDDEN_SIZE_2, HIDDEN_SIZE_3)
    weights_output = initialize_weights(HIDDEN_SIZE_3, OUTPUT_SIZE)
    
    biases_hidden1 = np.zeros(HIDDEN_SIZE_1)
    biases_hidden2 = np.zeros(HIDDEN_SIZE_2)
    biases_hidden3 = np.zeros(HIDDEN_SIZE_3)
    biases_output = np.zeros(OUTPUT_SIZE)
    
    learning_rate = INITIAL_LEARNING_RATE
    best_loss = float('inf')
    patience = 0
    no_improvement_count = 0
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i in range(num_samples):
            hidden1_output, hidden2_output, hidden3_output, final_output = forward_pass(
                inputs[i], weights_hidden1, weights_hidden2, weights_hidden3, weights_output,
                biases_hidden1, biases_hidden2, biases_hidden3, biases_output
            )
            sample_loss = cross_entropy_loss(final_output, targets[i])
            total_loss += sample_loss
            backpropagate(
                inputs[i], hidden1_output, hidden2_output, hidden3_output, final_output, targets[i],
                weights_hidden1, weights_hidden2, weights_hidden3, weights_output,
                biases_hidden1, biases_hidden2, biases_hidden3, biases_output, learning_rate
            )
        
        total_loss /= num_samples
        
        if total_loss < best_loss:
            no_improvement_count = 0
            best_loss = total_loss
            print(f"\033[32mEpoch {epoch + 1}: Loss improved to {total_loss:.6f}\033[0m")
        else:
            no_improvement_count += 1
            print(f"\033[33mEpoch {epoch + 1}: No improvement, patience {no_improvement_count}/{EARLY_STOPPING_PATIENCE}\033[0m")
            if no_improvement_count >= EARLY_STOPPING_PATIENCE:
                print(f"\033[31mEarly stopping at epoch {epoch + 1}\033[0m")
                break
        
        learning_rate = max(learning_rate * LEARNING_RATE_DECAY, MIN_LEARNING_RATE)
    
    print(f"\033[34mTraining complete. Best Loss: {best_loss:.6f}\033[0m")
    print(f"\033[34mNo Improvement Count: {no_improvement_count}\033[0m")
    
    # Save the trained model to file
    with open(MODEL_FILENAME, "wb") as file:
        pickle.dump({
            'weights_hidden1': weights_hidden1,
            'weights_hidden2': weights_hidden2,
            'weights_hidden3': weights_hidden3,
            'weights_output': weights_output,
            'biases_hidden1': biases_hidden1,
            'biases_hidden2': biases_hidden2,
            'biases_hidden3': biases_hidden3,
            'biases_output': biases_output
        }, file)
    print(f"\033[34mModel saved to '{MODEL_FILENAME}'\033[0m")

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility

    # Example XOR input and output
    inputs = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    targets = np.eye(OUTPUT_SIZE)

    train(inputs, targets, len(inputs))
  
