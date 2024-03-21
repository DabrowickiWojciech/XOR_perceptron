''' Program is made so that the weights are generated randomly
    and the output error is only treated like good or not without
    any sugestions in which way it should go'''
import numpy as np

''' Encapsuled perceptron class with their functions '''
class Perceptron:
    ''' Initializing perceptron variables '''
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        # First two values of weight are inputs and the last is bias weight
        self.weights = np.random.rand(input_size + 1)
        # Setting learning speed
        self.learning_rate = learning_rate
        # Setting how many epochs is needed to learn
        self.epochs = epochs

    ''' Activate function for neuron '''
    def activate(self, x):
        # Step function
        return 1 if x >= 0 else 0

    ''' Prediction function (main, estimating output) '''
    def predict(self, inputs):
        # Dot product of inserting inputs and their weights with biases
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        # Apply activation function of dot product
        return self.activate(summation)

    ''' Comparison function '''
    def train(self, training_inputs, targets):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, targets):
                # Uses prediction function to make a guess of outputs depending on weights
                prediction = self.predict(inputs)
                # Checks if prediction and targets are the same
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

def main():
    # Inputs and targets for XOR
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([0, 1, 1, 0])

    # Create and train Perceptron
    perceptron = Perceptron(input_size=2)
    perceptron.train(training_inputs, targets)

    # Test the trained Perceptron
    print("Testing trained Perceptron:")
    for inputs, label in zip(training_inputs, targets):
        prediction = perceptron.predict(inputs)
        print(f"Inputs: {inputs}, Predicted: {prediction}, Actual: {label}")

if __name__ == "__main__":
    main()