import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, momentum=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.loss_history = {'output': [], 'hidden': [], 'predict' : [], 'classification' : []}
        self.weights_history = {'W1': [], 'b1': [], 'W2': [], 'b2': []}
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)  # He initialization
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)  # He initialization
        self.b2 = np.zeros((1, output_size))
        
        # Initialize momentum terms
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        
        # Initialize adaptive learning rate terms
        self.epsilon = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.mW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.vW1_hat = np.zeros_like(self.W1)
        self.vb1_hat = np.zeros_like(self.b1)
        self.vW2_hat = np.zeros_like(self.W2)
        self.vb2_hat = np.zeros_like(self.b2)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_derivative(self, x):
        return x * (1 - x)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, input, label, output):
        self.delta2 = (label - output) * self.sigmoid_derivative(output)
        self.dW2 = np.dot(self.a1.T, self.delta2)
        self.db2 = np.sum(self.delta2)
        self.delta1 = np.dot(self.delta2, self.W2.T) * self.relu_derivative(self.a1)
        self.dW1 = np.dot(input.T, self.delta1)
        self.db1 = np.sum(self.delta1)
        return self.dW1, self.db1, self.dW2, self.db2
    
    def update_adaptive_learning_rate(self, dW1, db1, dW2, db2):
        # Update biased first and second moment estimates
        self.mW1 = self.beta1 * self.mW1 + (1 - self.beta1) * dW1
        self.mb1 = self.beta1 * self.mb1 + (1 - self.beta1) * db1
        self.mW2 = self.beta1 * self.mW2 + (1 - self.beta1) * dW2
        self.mb2 = self.beta1 * self.mb2 + (1 - self.beta1) * db2
        
        # Update unbiased first and second moment estimates
        self.vW1 = self.beta2 * self.vW1 + (1 - self.beta2) * np.square(dW1)
        self.vb1 = self.beta2 * self.vb1 + (1 - self.beta2) * np.square(db1)
        self.vW2 = self.beta2 * self.vW2 + (1 - self.beta2) * np.square(dW2)
        self.vb2 = self.beta2 * self.vb2 + (1 - self.beta2) * np.square(db2)
        
        # Compute bias-corrected first and second moment estimates
        self.mW1_hat = self.mW1 / (1 - self.beta1)
        self.mb1_hat = self.mb1 / (1 - self.beta1)
        self.mW2_hat = self.mW2 / (1 - self.beta1)
        self.mb2_hat = self.mb2 / (1 - self.beta1)
        self.vW1_hat = self.vW1 / (1 - self.beta2)
        self.vb1_hat = self.vb1 / (1 - self.beta2)
        self.vW2_hat = self.vW2 / (1 - self.beta2)
        self.vb2_hat = self.vb2 / (1 - self.beta2)
        
        # Update weights with adaptive learning rate
        self.W1 += self.learning_rate * self.mW1_hat / (np.sqrt(self.vW1_hat) + self.epsilon)
        self.b1 += self.learning_rate * self.mb1_hat / (np.sqrt(self.vb1_hat) + self.epsilon)
        self.W2 += self.learning_rate * self.mW2_hat / (np.sqrt(self.vW2_hat) + self.epsilon)
        self.b2 += self.learning_rate * self.mb2_hat / (np.sqrt(self.vb2_hat) + self.epsilon)
    
    def train(self, input, labels, epochs, batch_size=32):
        for _ in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(input.shape[0])
            input_shuffled = input[indices]
            labels_shuffled = labels[indices]
            
            # Mini-batch training
            for i in range(0, input.shape[0], batch_size):
                input_batch = input_shuffled[i:i+batch_size]
                labels_batch = labels_shuffled[i:i+batch_size]
                
                output = self.forward(input_batch)
                dW1, db1, dW2, db2 = self.backward(input_batch, labels_batch, output)
                self.update_adaptive_learning_rate(dW1, db1, dW2, db2)
            
            # Store weights at each epoch
            self.weights_history['W1'].append(self.W1.copy())
            self.weights_history['b1'].append(self.b1.copy())
            self.weights_history['W2'].append(self.W2.copy())
            self.weights_history['b2'].append(self.b2.copy())
            
            # Compute loss at hidden layer
            self.hidden_output = self.relu(np.dot(input, self.W1) + self.b1)
            self.hidden_loss = np.mean((self.hidden_output - np.mean(self.hidden_output, axis=0)) ** 2)
            self.loss_history['hidden'].append(self.hidden_loss)
            
            # Compute loss at output layer
            self.output = self.forward(input)
            self.output_loss = np.mean((labels - self.output) ** 2)
            self.loss_history['output'].append(self.output_loss)
            
            # Prediction loss
            self.loss = np.mean(np.abs(labels - self.output))
            self.loss_history['predict'].append(self.loss)
            
            # Classification error
            # Compute classification error
            self.predictions = (self.output > 0.5).astype(int)
            self.classification_error = np.mean(self.predictions != labels)
            self.loss_history['classification'].append(self.classification_error)
            
            # Checkign if error is sufficient
            if self.output_loss < 0.0001:
                break
        
    def predict(self, input):
        return self.forward(input)
    
    def plot(self):
        # Plot output error
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(model.loss_history['predict'], label='Predict Layer')
        plt.grid(visible=1)
        plt.title('Output error')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        # Plot MSE hidden error
        plt.subplot(2,2,2)
        plt.plot(max(model.loss_history['hidden']) - model.loss_history['hidden'], label='Hidden Layer')
        plt.grid(visible=1)
        plt.title('Hidden MSE layer error')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        # Plot MSE output error
        plt.subplot(2,2,3)
        plt.plot(model.loss_history['output'], label='Output Layer')
        plt.grid(visible=1)
        plt.title('Output MSE layer error')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        # Plot classification
        plt.subplot(2,2,4)
        plt.plot(model.loss_history['classification'], label='Output Layer')
        plt.grid(visible=1)
        plt.title('Classification')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')

        # Plot the changes in weights and biases over time
        plt.figure()
        for layer in ['W1', 'b1', 'W2', 'b2']:
            weights = np.array(model.weights_history[layer])
            for i in range(weights.shape[1]):
                plt.plot(weights[:, i], label=f'{layer}_{i+1}')
        plt.grid(visible=1)
        plt.title('Changes in Weights and Biases')
        plt.xlabel('Epoch')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.show()


# Define XOR inputs and labels
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])

# Initialize and train the neural network
input_size = 2
hidden_size = 2
output_size = 1
epochs = 100
batch_size = 32

# Initialize and train the neural network
model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(input, labels, epochs, batch_size)

# Test the trained model
predictions = model.predict(input)
print("Predictions:")
print(predictions)

# Plotting
model.plot()
