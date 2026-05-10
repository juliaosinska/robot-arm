import numpy as np
from config import L1, L2

# forward kinematics: given angles alpha and beta, calculate (x, y) position of the end effector
def forward_kinematics(alpha, beta):

    # first segment of the arm
    x1 = L1 * np.cos(alpha)
    y1 = L1 * np.sin(alpha)

    # second segment of the arm
    theta2 = alpha - beta

    x2 = x1 + L2 * np.cos(theta2)
    y2 = y1 + L2 * np.sin(theta2)

    return x2, y2