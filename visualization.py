import numpy as np
import matplotlib.pyplot as plt
from kinematics import forward_kinematics
from config import L1, L2
from matplotlib.animation import FuncAnimation

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


def plot_interactive_arm(model, X_max):

    fig, ax = plt.subplots(figsize=(8, 8))

    limit = L1 + L2

    resolution = 120

    x_space = np.linspace(-limit, limit, resolution)
    y_space = np.linspace(-limit, limit, resolution)

    xx, yy = np.meshgrid(x_space, y_space)

    points = np.column_stack((xx.ravel(), yy.ravel()))

    target = np.array([2.0, 2.0])

    # heatmap

    def compute_error_map(cursor):

        distances = np.sqrt(
            (points[:, 0] - cursor[0]) ** 2 +
            (points[:, 1] - cursor[1]) ** 2
        )

        points_norm = points / X_max

        predictions = model.predict(points_norm)

        alpha = predictions[:, 0] * np.pi
        beta = predictions[:, 1] * np.pi

        x_pred, y_pred = forward_kinematics(alpha, beta)

        errors = np.sqrt(
            (points[:, 0] - x_pred) ** 2 +
            (points[:, 1] - y_pred) ** 2
        )

        weighted = errors * (1 + distances * 0.1)

        return weighted.reshape(resolution, resolution)

    error_map = compute_error_map(target)

    image = ax.imshow(
        error_map,
        extent=[-limit, limit, -limit, limit],
        origin='lower',
        cmap='turbo'
    )

    plt.colorbar(image, ax=ax, label='Prediction error')


    arm_line, = ax.plot([], [], 'o-', lw=4)

    target_dot, = ax.plot([], [], 'rx', markersize=12)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_title('Interactive Robot Arm')

    ax.grid(True)


    def update_arm(cursor):

        x_norm = np.array([[cursor[0], cursor[1]]]) / X_max

        prediction = model.predict(x_norm)[0]

        alpha = prediction[0] * np.pi
        beta = prediction[1] * np.pi

        x1, y1, x2, y2 = forward_kinematics(alpha, beta, return_joints=True)

        arm_line.set_data(
            [0, x1, x2],
            [0, y1, y2]
        )

        target_dot.set_data([cursor[0]], [cursor[1]])

    update_arm(target)


    def on_mouse_move(event):

        if event.inaxes != ax:
            return

        target[0] = event.xdata
        target[1] = event.ydata

        update_arm(target)

        new_error_map = compute_error_map(target)

        image.set_data(new_error_map)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    plt.show()