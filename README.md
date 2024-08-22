# neural_network_in_python
---
Neural Networks: [in C Script](https://github.com/mdriyadkhan585/neural_network)
---
### Documentation for Neural Network Project

#### Overview

This documentation provides a comprehensive guide for using two Python scripts: `neural_network.py` for training a neural network model, and `run_inference.py` for running inference with a trained model. 

 ---

### Download Commands

If the URL of your repository is `https://github.com/username/repository.git`, you would use:

```bash
git clone https://github.com/username/repository.git
```
---


### Installing Dependencies

1. `requirements.txt`

This file lists all the Python packages your project depends on. You can create it with the following content:

```requirements
numpy
```

After cloning the repository, navigate into the project directory and install the dependencies listed in `requirements.txt`:

```bash
cd repository
pip install -r requirements.txt
```

---

## `neural_network.py` - Model Trainer

### Description

`neural_network.py` is responsible for defining, training, and saving a neural network model. This script includes:
- Model architecture and activation functions
- Weight initialization
- Forward pass and backpropagation algorithms
- Training loop with loss calculation and early stopping
- Model saving

### Key Functions

1. **`initialize_weights(size)`**
   - **Purpose**: Initializes weights for the neural network layers.
   - **Input**: `size` (int) - Number of weights to initialize.
   - **Output**: `np.ndarray` - Array of initialized weights.

2. **`relu(x)`**
   - **Purpose**: Applies the ReLU activation function.
   - **Input**: `x` (float) - Input value.
   - **Output**: `float` - Activated value.

3. **`elu(x)`**
   - **Purpose**: Applies the ELU activation function.
   - **Input**: `x` (float) - Input value.
   - **Output**: `float` - Activated value.

4. **`sigmoid(x)`**
   - **Purpose**: Applies the sigmoid activation function.
   - **Input**: `x` (float) - Input value.
   - **Output**: `float` - Activated value.

5. **`swish(x)`**
   - **Purpose**: Applies the Swish activation function.
   - **Input**: `x` (float) - Input value.
   - **Output**: `float` - Activated value.

6. **`softmax(output)`**
   - **Purpose**: Applies the softmax function to the output layer.
   - **Input**: `output` (np.ndarray) - Array of logits.
   - **Output**: `np.ndarray` - Softmax probabilities.

7.
```python
forward_pass(input, weights_hidden1, weights_hidden2, weights_hidden3, weights_output, biases_hidden1, biases_hidden2, biases_hidden3, biases_output)
```
   - **Purpose**: Performs the forward pass of the network.
   - **Inputs**: 
     - `input` (np.ndarray) - Input data.
     - `weights_hidden1`, `weights_hidden2`, `weights_hidden3`, `weights_output` (np.ndarray) - Weights for each layer.
     - `biases_hidden1`, `biases_hidden2`, `biases_hidden3`, `biases_output` (np.ndarray) - Biases for each layer.
   - **Outputs**: 
     - `hidden1_output`, `hidden2_output`, `hidden3_output`, `final_output` (np.ndarray) - Output of each layer.

  9.
```python
backpropagate(input, hidden1_output, hidden2_output, hidden3_output, final_output, actual_output, weights_hidden1, weights_hidden2, weights_hidden3, weights_output, biases_hidden1, biases_hidden2, biases_hidden3, biases_output, learning_rate)
```
   - **Purpose**: Updates weights and biases through backpropagation.
   - **Inputs**:
     - `input`, `hidden1_output`, `hidden2_output`, `hidden3_output`, `final_output` (np.ndarray) - Layer outputs.
     - `actual_output` (np.ndarray) - True output values.
     - `weights_hidden1`, `weights_hidden2`, `weights_hidden3`, `weights_output` (np.ndarray) - Weights to be updated.
     - `biases_hidden1`, `biases_hidden2`, `biases_hidden3`, `biases_output` (np.ndarray) - Biases to be updated.
     - `learning_rate` (float) - Learning rate for updates.
   - **Output**: None.

10.
```python
    train(inputs, targets, num_samples)
```
   - **Purpose**: Trains the neural network model.
   - **Inputs**:
     - `inputs` (np.ndarray) - Input data.
     - `targets` (np.ndarray) - Target output data.
     - `num_samples` (int) - Number of training samples.
   - **Output**: None (Trains and saves the model).

### How to Use

1. **Prepare Data**: Organize your input data and target labels.
2. **Configure Parameters**: Adjust the hyperparameters and model architecture if needed.
3. **Run Training**:
   ```bash
   python neural_network.py
   ```
   This will train the model on the provided data and save it to `trained_model.pkl`.

### Example

Here's an example of the data preparation and training:
```python
import numpy as np

# Sample data
inputs = np.random.rand(10, INPUT_SIZE)  # Replace with actual data
targets = np.eye(OUTPUT_SIZE)[np.random.randint(0, OUTPUT_SIZE, 10)]  # Example one-hot encoded targets

# Train the model
train(inputs, targets, len(inputs))
```

---

## `run_inference.py` - Run Inference

### Description

`run_inference.py` is used to load a pre-trained model and perform inference on new data. This script includes:
- Model loading from a file
- Forward pass for inference

### Key Functions

1.
```python
load_model(weights_hidden1, weights_hidden2, weights_hidden3, weights_output, biases_hidden1, biases_hidden2, biases_hidden3, biases_output)
```
   - **Purpose**: Loads model weights and biases from a file.
   - **Inputs**:
     - `weights_hidden1`, `weights_hidden2`, `weights_hidden3`, `weights_output` (np.ndarray) - Arrays to store weights.
     - `biases_hidden1`, `biases_hidden2`, `biases_hidden3`, `biases_output` (np.ndarray) - Arrays to store biases.
   - **Output**: None.

3.
```python
forward_pass(input, weights_hidden1, weights_hidden2, weights_hidden3, weights_output, biases_hidden1, biases_hidden2, biases_hidden3, biases_output, hidden1_output, hidden2_output, hidden3_output, final_output)
```
   - **Purpose**: Performs inference by running a forward pass with the provided input.
   - **Inputs**:
     - `input` (np.ndarray) - Input data for inference.
     - `weights_hidden1`, `weights_hidden2`, `weights_hidden3`, `weights_output` (np.ndarray) - Loaded weights.
     - `biases_hidden1`, `biases_hidden2`, `biases_hidden3`, `biases_output` (np.ndarray) - Loaded biases.
   - **Outputs**:
     - `hidden1_output`, `hidden2_output`, `hidden3_output`, `final_output` (np.ndarray) - Output of each layer.

### How to Use

1. **Load the Model**:
   Ensure that `trained_model.pkl` is available in the same directory.
   
2. **Run Inference**:
   ```bash
   python run_inference.py
   ```

### Example

Here’s how to use the `run_inference.py` script to perform inference:
```python
import numpy as np

# Load the trained model and perform inference
test_input = np.random.rand(INPUT_SIZE)  # Replace with actual test data
forward_pass(test_input, weights_hidden1, weights_hidden2, weights_hidden3, weights_output,
             biases_hidden1, biases_hidden2, biases_hidden3, biases_output,
             hidden1_output, hidden2_output, hidden3_output, final_output)

print("Inference result:")
for i in range(OUTPUT_SIZE):
    print(f"Class {i}: {final_output[i]:.4f}")
```

---

### Summary

- **`neural_network.py`**: For training the model and saving it.
- **`run_inference.py`**: For loading the trained model and running inference.

Ensure that all dependencies (e.g., NumPy) are installed and the model file (`trained_model.pkl`) is available when running inference.

---
