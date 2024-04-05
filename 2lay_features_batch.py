import numpy as np
import matplotlib.pyplot as plt

class TwoLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, momentum=0.9, initial_learning_rate=0.2, learning_rate_decay=0.9):
        # Model parameters
        self.hidden_weights = np.random.rand(input_size, hidden_size)
        self.hidden_bias = np.random.rand(hidden_size)
        self.output_weights = np.random.rand(hidden_size, output_size)
        self.output_bias = np.random.rand(output_size)
        
        # Lists to store error metrics and weights for visualization
        self.plt_errors = []
        self.plt_mse_errors_output = []
        self.plt_mse_errors_hidden = []
        self.plt_weights_hidden_first = []
        self.plt_weights_hidden_second = []
        self.plt_weights_hidden_third = []
        self.plt_weights_hidden_fourth = []
        self.plt_weights_output_first = []
        self.plt_weights_output_second = []
        self.plt_classification_error = []
        
        # Momentum parameters
        self.momentum = momentum
        self.prev_output_weight_update = np.zeros_like(self.output_weights)
        self.prev_output_bias_update = np.zeros_like(self.output_bias)
        self.prev_hidden_weight_update = np.zeros_like(self.hidden_weights)
        self.prev_hidden_bias_update = np.zeros_like(self.hidden_bias)
        
        # Learning rate parameters
        self.learning_rate = initial_learning_rate
        self.learning_rate_decay = learning_rate_decay

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        # Derivative of sigmoid activation function
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def predict(self, inputs):
        # Forward pass through the network
        self.hidden_layer_input = np.dot(inputs, self.hidden_weights) + self.hidden_bias  # Input to hidden layer
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)  # Activation of hidden layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.output_weights) + self.output_bias  # Input to output layer
        self.output = self.sigmoid(self.output_layer_input)  # Activation of output layer
        return self.output, self.hidden_layer_output

    def train(self, inputs, targets, rewards, batch_size=1, learning_rate=0.2, max_epochs=3000):
        # Training the neural network
        for _ in range(max_epochs):
            for batch_start in range(0, len(inputs), batch_size):
                # Iterate through batches
                batch_inputs = inputs[batch_start:batch_start+batch_size]  # Batch of input data
                batch_targets = targets[batch_start:batch_start+batch_size]  # Batch of target outputs
                self.output, self.hidden_layer_output = self.predict(batch_inputs)  # Forward pass
                self.output_error = batch_targets - self.output  # Error at the output layer
                self.output_delta = self.output_error * self.derivative_sigmoid(self.output_layer_input)  # Delta at the output layer
                self.hidden_error = np.dot(self.output_delta, self.output_weights.T)  # Error at the hidden layer
                self.hidden_delta = self.hidden_error * self.derivative_sigmoid(self.hidden_layer_input)  # Delta at the hidden layer
                
                # Update weights and biases
                output_weight_gradient = np.dot(self.hidden_layer_output.T, self.output_delta)
                self.prev_output_weight_update = learning_rate * output_weight_gradient + self.momentum * self.prev_output_weight_update
                self.output_weights += self.prev_output_weight_update
                self.prev_output_bias_update = learning_rate * np.sum(self.output_delta, axis=0) + self.momentum * self.prev_output_bias_update
                self.output_bias += self.prev_output_bias_update

                hidden_weight_gradient = np.dot(batch_inputs.T, self.hidden_delta)
                self.prev_hidden_weight_update = learning_rate * hidden_weight_gradient + self.momentum * self.prev_hidden_weight_update
                self.hidden_weights += self.prev_hidden_weight_update
                self.prev_hidden_bias_update = learning_rate * np.sum(self.hidden_delta, axis=0) + self.momentum * self.prev_hidden_bias_update
                self.hidden_bias += self.prev_hidden_bias_update

                # Apply reinforcement learning
                self.reward_multiplier = rewards[batch_start:batch_start+batch_size] * self.output_error
                self.output_weights += learning_rate * np.dot(self.hidden_layer_output.T, self.reward_multiplier)
                self.output_bias += learning_rate * np.sum(self.reward_multiplier)

            # Classify outputs
            self.output[self.output >= 0.5] = 1
            self.output[self.output < 0.5] = 0
            self.classification_error = np.mean(np.abs(self.output[:batch_size] - batch_targets))

            # Adjust learning rate based on error change
            if len(self.plt_errors) > 1 and self.plt_errors[-1] >= self.plt_errors[-2]:
                self.learning_rate *= self.learning_rate_decay

            # Error metrics
            self.error = np.mean(np.abs(self.output_error))
            self.plt_errors.append(self.error)

            self.mse_error_output = np.mean(self.output_error ** 2)
            self.plt_mse_errors_output.append(self.mse_error_output)
            self.mse_error_hidden = np.mean(self.hidden_error ** 2)
            self.plt_mse_errors_hidden.append(self.mse_error_hidden)

            # Store weights for visualization
            self.plt_weights_hidden_first.append(self.hidden_weights[0][0])
            self.plt_weights_hidden_second.append(self.hidden_weights[0][1])
            self.plt_weights_hidden_third.append(self.hidden_weights[1][0])
            self.plt_weights_hidden_fourth.append(self.hidden_weights[1][1])
            self.plt_weights_output_first.append(self.output_weights[0][0])
            self.plt_weights_output_second.append(self.output_weights[1][0])

            self.plt_classification_error.append(self.classification_error)

            # Termination condition
            if self.mse_error_output < 0.0001:
                break



    def plot(self):
        # Visualization of error metrics and weights
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(self.plt_errors)
        plt.title('Output error')
        plt.grid(visible=1)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        
        plt.subplot(2,2,2)
        plt.plot(self.plt_mse_errors_output)
        plt.title('Mean square output error')
        plt.grid(visible=1)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        
        plt.subplot(2,2,3)
        plt.plot(self.plt_mse_errors_hidden)
        plt.title('Mean square hidden layer error')
        plt.grid(visible=1)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        
        plt.subplot(2,2,4)
        plt.plot(self.plt_classification_error)
        plt.title('Classification error')
        plt.grid(visible=1)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.draw()
        
        plt.figure()
        plt.plot(self.plt_weights_output_first)
        plt.plot(self.plt_weights_output_second)
        plt.plot(self.plt_weights_hidden_first)
        plt.plot(self.plt_weights_hidden_second)
        plt.plot(self.plt_weights_hidden_third)
        plt.plot(self.plt_weights_hidden_fourth)
        plt.title('Output layer weights')
        plt.grid(visible=1)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        
        plt.show()

def main():
    # Input data, target outputs, and rewards
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    rewards = np.array([[4], [4], [4], [4]])

    # Create and train the neural network
    perceptron = TwoLayerPerceptron(input_size=2, hidden_size=2, output_size=1)
    perceptron.train(inputs, targets, rewards, batch_size=2)
    perceptron.plot()  # Visualize training progress
    
    # Test the trained neural network
    print("\nTesting XOR gate:")
    for i in range(len(inputs)):
        x1 = inputs[i][0]
        x2 = inputs[i][1]
        prediction = perceptron.predict(np.array([[x1, x2]]))[0]
        print(f"Input: ({x1}, {x2}), Predicted Output: {prediction}")

if __name__ == "__main__":
    main()
