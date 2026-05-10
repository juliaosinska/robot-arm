import numpy as np
from config import *
from dataset import generate_dataset
from model import create_model
from train import train_model
from visualization import plot_training, plot_error_map

# generating dataset
X, Y = generate_dataset()

# normalization
X_max = np.max(np.abs(X), axis=0)
X = X / X_max

Y = Y / np.pi

# creating model
model = create_model()

# training
history, X_test, Y_test = train_model(
    model,
    X,
    Y,
    EPOCHS,
    BATCH_SIZE
)

# visualization
plot_training(history)
plot_error_map(model, X_max)