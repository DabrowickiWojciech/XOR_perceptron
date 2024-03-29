''' Program is made so that the weights are generated randomly
    and the output error is only treated like good or not without
    any sugestions in which way it should go, but there are 
    2 hidden layers and ReLU function activation'''
import numpy as np

''' Encapsuled perceptron class with their functions '''
class TwoLayerPerceptron:
    ''' Initializing perceptron variables '''
    def __init__(self, input_size, hidden_size, output_size):
        ''' First layer '''
        # Setting weight and biases for two layers of perceptron
        self.hidden_weights = np.random.rand(input_size, hidden_size)
        self.hidden_bias = np.random.rand(hidden_size)
        ''' Second layer '''
        self.output_weights = np.random.rand(hidden_size, output_size)
        self.output_bias = np.random.rand(output_size)

    ''' Activate function for neuron '''
    # ReLU activation function
    def ReLU(self, x):
        self.x = x
        return np.maximum(0, x)

    def predict(self, inputs):
        self.inputs = inputs
        # Dot product of inserting inputs and their weights with biases
        self.hidden_layer_input = np.dot(self.inputs, self.hidden_weights) + self.hidden_bias
        # Applying activation function
        self.hidden_layer_output = self.ReLU(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.output_weights) + self.output_bias
        self.output = self.ReLU(self.output_layer_input)
        # Returning estimated values depending on two layers
        return self.output, self.hidden_layer_output

    ''' Comparison function '''
    def train(self, inputs, targets, learning_rate=0.1, max_epochs=10000):
        self.inputs = inputs
        self.targets = targets
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        for _ in range(max_epochs):
            # Uses prediction function to make a guess of outputs depending on weights
            self.output, self.hidden_layer_oputput = self.predict(self.inputs)

            # Backpropagation
            self.output_error = self.targets - self.output
            self.output_delta = self.output_error
            self.hidden_error = np.dot(self.output_delta, self.output_weights.T)
            self.hidden_delta = self.hidden_error * (self.hidden_layer_output > 0)

            # Update weights and biases
            self.output_weights += self.learning_rate * np.dot(self.hidden_layer_output.T, self.output_delta)
            self.output_bias += self.learning_rate * np.sum(self.output_delta, axis = 0)
            self.hidden_weights += self.learning_rate * np.dot(inputs.T, self.hidden_delta)
            self.hidden_bias += self.learning_rate * np.sum(self.hidden_delta, axis = 0)

            # Check for convergence
            if np.mean(np.abs(self.output_error)) < 0.01:
                break

def main():
    # Inputs and targets for XOR
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # Create two-layered perceptron
    perceptron = TwoLayerPerceptron(input_size=2, hidden_size=2, output_size=1)

    # Train perceptron
    perceptron.train(inputs, targets)

    # Test the trained perceptron
    print("\nTesting XOR gate:")
    for i in range(len(inputs)):
        x1 = inputs[i][0]
        x2 = inputs[i][1]
        prediction = perceptron.predict(np.array([[x1, x2]]))[0]
        print(f"Input: ({x1}, {x2}), Predicted Output: {prediction}")

if __name__ == "__main__":
    main()