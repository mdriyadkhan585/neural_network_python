import numpy as np
import pickle
import math
from scipy.special import expit  # For stable sigmoid computation

# Constants
INPUT_SIZE = 10
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
HIDDEN_SIZE_3 = 32
OUTPUT_SIZE = 10
MODEL_FILENAME = "trained_model.pkl"

# Color Codes for Output
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def elu(x):
    return np.where(x > 0, x, np.exp(x) - 1)

def sigmoid(x):
    return expit(x)

def swish(x):
    return x * sigmoid(x)

def softmax(output):
    max_val = np.max(output)
    exp_vals = np.exp(output - max_val)
    return exp_vals / np.sum(exp_vals)

# Load Model from File
def load_model():
    try:
        with open(MODEL_FILENAME, "rb") as file:
            weights_hidden1 = np.fromfile(file, dtype=np.float64, count=INPUT_SIZE * HIDDEN_SIZE_1).reshape(INPUT_SIZE, HIDDEN_SIZE_1)
            weights_hidden2 = np.fromfile(file, dtype=np.float64, count=HIDDEN_SIZE_1 * HIDDEN_SIZE_2).reshape(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
            weights_hidden3 = np.fromfile(file, dtype=np.float64, count=HIDDEN_SIZE_2 * HIDDEN_SIZE_3).reshape(HIDDEN_SIZE_2, HIDDEN_SIZE_3)
            weights_output = np.fromfile(file, dtype=np.float64, count=HIDDEN_SIZE_3 * OUTPUT_SIZE).reshape(HIDDEN_SIZE_3, OUTPUT_SIZE)
            biases_hidden1 = np.fromfile(file, dtype=np.float64, count=HIDDEN_SIZE_1)
            biases_hidden2 = np.fromfile(file, dtype=np.float64, count=HIDDEN_SIZE_2)
            biases_hidden3 = np.fromfile(file, dtype=np.float64, count=HIDDEN_SIZE_3)
            biases_output = np.fromfile(file, dtype=np.float64, count=OUTPUT_SIZE)
        
        print(COLOR_GREEN + f"Model loaded successfully from '{MODEL_FILENAME}'." + COLOR_RESET)
        return (weights_hidden1, weights_hidden2, weights_hidden3, weights_output,
                biases_hidden1, biases_hidden2, biases_hidden3, biases_output)
    
    except FileNotFoundError:
        print(COLOR_RED + f"Failed to load model from '{MODEL_FILENAME}'." + COLOR_RESET)
        raise

# Forward Pass for Inference
def forward_pass(input, weights_hidden1, weights_hidden2, weights_hidden3, weights_output,
                  biases_hidden1, biases_hidden2, biases_hidden3, biases_output):
    
    # Hidden Layer 1
    hidden1_output = np.dot(input, weights_hidden1) + biases_hidden1
    hidden1_output = swish(hidden1_output)
    
    # Hidden Layer 2
    hidden2_output = np.dot(hidden1_output, weights_hidden2) + biases_hidden2
    hidden2_output = relu(hidden2_output)
    
    # Hidden Layer 3
    hidden3_output = np.dot(hidden2_output, weights_hidden3) + biases_hidden3
    hidden3_output = elu(hidden3_output)
    
    # Output Layer
    final_output = np.dot(hidden3_output, weights_output) + biases_output
    final_output = softmax(final_output)
    
    return final_output

# Main Function for Inference
if __name__ == "__main__":
    weights_hidden1, weights_hidden2, weights_hidden3, weights_output, \
    biases_hidden1, biases_hidden2, biases_hidden3, biases_output = load_model()

    # Example input for testing
    test_input = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # Modify as needed

    final_output = forward_pass(test_input, weights_hidden1, weights_hidden2, weights_hidden3, weights_output,
                                 biases_hidden1, biases_hidden2, biases_hidden3, biases_output)

    print("Inference result:")
    for i in range(OUTPUT_SIZE):
        print(f"Class {i}: {final_output[i]:.4f}")
      
