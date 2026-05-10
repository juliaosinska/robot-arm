import numpy as np
import matplotlib.pyplot as plt
from kinematics import forward_kinematics

def plot_training(history):

    plt.figure(figsize=(8, 5))

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')

    plt.xlabel('Epoka')
    plt.ylabel('MSE')
    plt.title('Błąd uczenia sieci')
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_error_map(model, X_max):

    # map 120 x 120 points in the workspace of the robot arm
    resolution = 120

    # 8 and -8 because the arm can reach 8 units in any direction (L1 + L2 = 8)
    x_space = np.linspace(-8, 8, resolution)
    y_space = np.linspace(-8, 8, resolution)

    xx, yy = np.meshgrid(x_space, y_space)

    # stacking the grid points into a 2D array of shape
    points = np.column_stack((xx.ravel(), yy.ravel()))

    # normalization of the points to be between 0 and 1 (same as the output of the model)
    points_norm = points / X_max

    # predicting the angles for each point in the workspace using the trained model
    predictions = model.predict(points_norm, verbose=0)

    # converting the predicted angles back to radians (0 to pi)
    alpha = predictions[:, 0] * np.pi
    beta = predictions[:, 1] * np.pi

    # checking where the arm would be for each predicted angle
    x_pred, y_pred = forward_kinematics(alpha, beta)

    # checking the distance between the predicted position of the arm and the actual point in the workspace
    errors = np.sqrt(
        (points[:, 0] - x_pred) ** 2 +
        (points[:, 1] - y_pred) ** 2
    )

    # reshaping the error array to match the grid shape for visualization
    error_map = errors.reshape(resolution, resolution)

    plt.figure(figsize=(8, 8))

    # heatmap of the error across the workspace where warmer colors indicate higher errors
    image = plt.imshow(
        error_map,
        extent=[-8, 8, -8, 8],
        origin='lower',
        cmap='turbo'
    )

    plt.colorbar(image, label='Błąd odległości')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.title('Mapa błędu ramienia robota')

    plt.tight_layout()
    plt.show()