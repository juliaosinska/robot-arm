import numpy as np
import matplotlib.pyplot as plt
from kinematics import forward_kinematics


# =====================================
# WYKRES BŁĘDU
# =====================================

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


# =====================================
# MAPA DOKŁADNOŚCI
# =====================================

def plot_error_map(model, X_max):

    resolution = 120

    x_space = np.linspace(-8, 8, resolution)
    y_space = np.linspace(-8, 8, resolution)

    xx, yy = np.meshgrid(x_space, y_space)

    # Tworzymy listę punktów
    points = np.column_stack((xx.ravel(), yy.ravel()))

    # Normalizacja
    points_norm = points / X_max

    # JEDNA predykcja dla wszystkich punktów
    predictions = model.predict(points_norm, verbose=0)

    alpha = predictions[:, 0] * np.pi
    beta = predictions[:, 1] * np.pi

    # Kinematyka prosta
    x_pred, y_pred = forward_kinematics(alpha, beta)

    # Liczenie błędu
    errors = np.sqrt(
        (points[:, 0] - x_pred) ** 2 +
        (points[:, 1] - y_pred) ** 2
    )

    # Zamiana na macierz 2D
    error_map = errors.reshape(resolution, resolution)

    # =====================================
    # RYSOWANIE
    # =====================================

    plt.figure(figsize=(8, 8))

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