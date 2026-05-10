from sklearn.model_selection import train_test_split


def train_model(model, X, Y, epochs, batch_size):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    # additional check if the error is decreasing during validation (same as training)
    history = model.fit(
        X_train,
        Y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return history, X_test, Y_test