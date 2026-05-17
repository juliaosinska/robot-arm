import numpy as np

# activation functions and their derivatives
def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(z, _output=None):
    return (z > 0).astype(float)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(_z, output):
    return output * (1.0 - output)


def linear(x):
    return x


def linear_derivative(_z, _output=None):
    return np.ones_like(_z)


def mse_loss(y_pred, y_true):
    diff = y_pred - y_true
    return np.mean(np.sum(diff**2, axis=1))


def mse_loss_grad(y_pred, y_true):
    return 2.0 * (y_pred - y_true) / y_true.shape[0]


# history class to store training history and metrics
class History:
    def __init__(self, history):
        self.history = history


class DenseLayer:
    def __init__(self, input_dim, output_dim, activation='linear', adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.activation = self._get_activation(activation)
        self.activation_grad = self._get_activation_grad(activation)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, size=(input_dim, output_dim))
        self.bias = np.zeros(output_dim, dtype=float)
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)
        self.input = None
        self.z = None
        self.output = None


    def _get_activation(self, name):
        if name == 'relu':
            return relu
        
        if name == 'sigmoid':
            return sigmoid
        
        return linear


    def _get_activation_grad(self, name):
        if name == 'relu':
            return relu_derivative
        
        if name == 'sigmoid':
            return sigmoid_derivative
        
        return linear_derivative


    def forward(self, inputs):
        self.input = np.asarray(inputs, dtype=float)
        self.z = self.input @ self.weights + self.bias
        self.output = self.activation(self.z)

        return self.output


    def backward(self, gradient, learning_rate, timestep):
        activation_grad = self.activation_grad(self.z, self.output)
        delta = gradient * activation_grad
        grad_w = self.input.T @ delta
        grad_b = np.sum(delta, axis=0)
        grad_input = delta @ self.weights.T
        
        self.m_w = self.adam_beta1 * self.m_w + (1 - self.adam_beta1) * grad_w
        self.v_w = self.adam_beta2 * self.v_w + (1 - self.adam_beta2) * grad_w**2
        self.m_b = self.adam_beta1 * self.m_b + (1 - self.adam_beta1) * grad_b
        self.v_b = self.adam_beta2 * self.v_b + (1 - self.adam_beta2) * grad_b**2
        
        m_w_hat = self.m_w / (1 - self.adam_beta1**timestep)
        v_w_hat = self.v_w / (1 - self.adam_beta2**timestep)
        m_b_hat = self.m_b / (1 - self.adam_beta1**timestep)
        v_b_hat = self.v_b / (1 - self.adam_beta2**timestep)
        
        self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.adam_eps)
        self.bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.adam_eps)
        
        return grad_input


class NeuralNetwork:
    def __init__(self, input_dim, layer_sizes, activations=None, learning_rate=0.001):
        layer_sizes = list(layer_sizes)

        if activations is None:
            activations = ['relu'] * len(layer_sizes)
            activations[-1] = 'sigmoid'

        if len(activations) != len(layer_sizes):
            raise ValueError('activations length must match layer_sizes length')


        self.layers = []
        current_dim = input_dim

        for size, activation in zip(layer_sizes, activations):
            self.layers.append(DenseLayer(current_dim, size, activation))
            current_dim = size

        self.learning_rate = learning_rate


    def forward(self, inputs):
        output = np.asarray(inputs, dtype=float)

        for layer in self.layers:
            output = layer.forward(output)

        return output


    def predict(self, inputs, verbose=0):
        return self.forward(inputs)


    def fit(self, X, Y, validation_split=0.0, epochs=1, batch_size=32, verbose=1):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n_samples = X.shape[0]
        history = {'loss': [], 'val_loss': []}

        if validation_split > 0.0:
            val_count = int(n_samples * validation_split)

            if val_count == 0:
                raise ValueError('validation_split is too small for the dataset size')
            
            permutation = np.random.permutation(n_samples)
            X = X[permutation]
            Y = Y[permutation]
            X_val = X[:val_count]
            Y_val = Y[:val_count]
            X_train = X[val_count:]
            Y_train = Y[val_count:]

        else:
            X_train = X
            Y_train = Y
            X_val = None
            Y_val = None

        for epoch in range(1, epochs + 1):
            permutation = np.random.permutation(X_train.shape[0])
            X_train = X_train[permutation]
            Y_train = Y_train[permutation]

            timestep = 0

            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]
                predictions = self.forward(X_batch)
                gradient = mse_loss_grad(predictions, Y_batch)
                timestep += 1

                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.learning_rate, timestep)

            train_loss = mse_loss(self.predict(X_train), Y_train)
            history['loss'].append(train_loss)

            if X_val is not None:
                val_loss = mse_loss(self.predict(X_val), Y_val)
                history['val_loss'].append(val_loss)
            else:
                history['val_loss'].append(train_loss)

            if verbose:
                message = f'Epoch {epoch}/{epochs} - loss: {train_loss:.6f}'
                if X_val is not None:
                    message += f" - val_loss: {history['val_loss'][-1]:.6f}"
                print(message)

        return History(history)


def create_model(
    input_dim=2,
    layer_sizes=(64, 64, 32, 2),
    activations=None,
    learning_rate=0.001
):
    
    if activations is None:
        activations = ['relu', 'relu', 'relu', 'sigmoid']

    return NeuralNetwork(input_dim=input_dim, layer_sizes=layer_sizes, activations=activations, learning_rate=learning_rate)
