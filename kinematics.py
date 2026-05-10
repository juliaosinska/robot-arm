import numpy as np
from config import L1, L2

# forward kinematics: given angles alpha and beta, calculate (x, y) position of the end effector
def forward_kinematics(alpha, beta):

    # arm stuck in (0, 0) position and we want to calculate the position of the end of the first segment
    x1 = L1 * np.cos(alpha)
    y1 = L1 * np.sin(alpha)

    # second segment of the arm (L2) is attached to the end of the first segment
    # and its angle is relative to the first segment
    theta2 = alpha - beta

    # we stick the second segment to the end of the first segment and calculate its position
    x2 = x1 + L2 * np.cos(theta2)
    y2 = y1 + L2 * np.sin(theta2)

    return x2, y2