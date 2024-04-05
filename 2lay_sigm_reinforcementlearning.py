import numpy as np
import matplotlib.pyplot as plt

''' Encapsuled perceptron class with their functions '''
class TwoLayerPerceptron:
    ''' Initializing perceptron variables '''
    def __init__(self, input_size, hidden_size, output_size):
        # Setting weight and biases for two layers of perceptron
        ''' First layer '''
        self.hidden_weights = np.random.rand(input_size, hidden_size)
        self.hidden_bias = np.random.rand(hidden_size)
        ''' Second layer '''
        self.output_weights = np.random.rand(hidden_size, output_size)
        self.output_bias = np.random.rand(output_size)
        self.plt_errors = []

    ''' Activate function for neuron '''
    # Sigmoid activation function
    def sigmoid(self, x):
        self.x = x
        return 1 / (1 + np.exp(-self.x))

    ''' Function necessary for backpropagation algorithm performance '''
    def derivative_sigmoid(self, x):
        self.x = x
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    ''' Prediction function (main, estimating output) '''
    def predict(self, inputs):
        self.inputs = inputs
        # Dot product of inserting inputs and their weights with biases
        self.hidden_layer_input = np.dot(self.inputs, self.hidden_weights) + self.hidden_bias
        # Applying activation function
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.output_weights) + self.output_bias
        self.output = self.sigmoid(self.output_layer_input)
        # Returning estimated values depending on two layers
        return self.output, self.hidden_layer_output

    ''' Comparison function '''
    def train(self, inputs, targets, rewards, learning_rate=0.1, max_epochs=3000):
        self.inputs = inputs
        self.targets = targets
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.rewards = rewards
        for _ in range(self.max_epochs):
            # Uses prediction function to make a guess of outputs depending on weights
            self.output, self.hidden_layer_oputput = self.predict(self.inputs)

            # Checks if prediction and targets are the same
            self.output_error = self.targets - self.output
            self.output_delta = self.output_error * self.derivative_sigmoid(self.output_layer_input)
            self.hidden_error = np.dot(self.output_delta, self.output_weights.T)
            self.hidden_delta = self.hidden_error * self.derivative_sigmoid(self.hidden_layer_input)

            # Update weights and biases
            self.output_weights += self.learning_rate * np.dot(self.hidden_layer_output.T, self.output_delta)         
            self.output_bias += self.learning_rate * np.sum(self.output_delta, axis = 0)            
            self.hidden_weights += self.learning_rate * np.dot(self.inputs.T, self.hidden_delta)
            self.hidden_bias += self.learning_rate * np.sum(self.hidden_delta, axis = 0)

            # Storing data for plot
            self.error = np.mean(np.abs(self.output_error))
            self.plt_errors.append(self.error)

            # Check for convergence
            if self.error < 0.01:
                break

            # Apply reinforcement learning
            self.reward_multiplier = self.rewards * self.output_error
            self.output_weights += self.learning_rate * np.dot(self.hidden_layer_output.T, self.reward_multiplier)
            self.output_bias += self.learning_rate * np.sum(self.reward_multiplier)

    def plot_errors(self):
        plt.plot(self.plt_errors)
        plt.title('Error during Training with backpropagation and reinforcement')
        plt.grid(visible=1)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.show()

''' Main function '''
def main():
    # Inputs and targets for XOR
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    rewards = np.array([[8], [8], [8], [8]])  # All correct predictions initially

    # Create two-layered perceptron
    perceptron = TwoLayerPerceptron(input_size=2, hidden_size=2, output_size=1)

    # Train perceptron with reinforcement learning
    perceptron.train(inputs, targets, rewards)

    # Ploting
    perceptron.plot_errors()
    
    # Test the trained perceptron
    print("\nTesting XOR gate:")
    for i in range(len(inputs)):
        x1 = inputs[i][0]
        x2 = inputs[i][1]
        prediction = perceptron.predict(np.array([[x1, x2]]))[0]
        print(f"Input: ({x1}, {x2}), Predicted Output: {prediction}")

if __name__ == "__main__":
    main()
