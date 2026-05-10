import numpy as np
from config import SAMPLES
from kinematics import forward_kinematics

# generating random dataset of (x, y) and (alpha, beta) pairs
def generate_dataset():

    alpha = np.random.uniform(0, np.pi, SAMPLES)
    beta = np.random.uniform(0, np.pi, SAMPLES)

    x, y = forward_kinematics(alpha, beta)

    X = np.column_stack((x, y))
    Y = np.column_stack((alpha, beta))

    return X, Y