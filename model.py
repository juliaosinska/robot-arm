from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_model():
    model = Sequential([
        # 3 hidden layers beacause the problem is non-linear
        # and we want to capture complex relationships between angles and positions
        # relu keeps the output of each layer non-negative
        Dense(64, activation='relu', input_shape=(2,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        # output layer with 2 neurons (x and y) and sigmoid activation to keep the output between 0 and 1
        Dense(2, activation='sigmoid')
    ])

    model.compile(
        # adam - optimization algorithm that adapts the learning rate during training
        optimizer='adam',
        # mean squared error loss function - the further the predicted position is from the true position
        # the higher the punishment
        loss='mse',
        # (additional) mean absolute error metric - how much in avg the predicted position is off from the true position
        metrics=['mae']
    )

    return model