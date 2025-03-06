import tensorflow as tf
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam, SGD, RMSprop # type: ignore

def build_ann(input_shape, num_layers=2, neurons_per_layer=32, activation="ReLU", dropout_rate=0.2, optimizer="Adam", learning_rate=0.001):
    """
    Builds an Artificial Neural Network (ANN) model with given hyperparameters.

    Args:
        input_shape (int): Number of input features.
        num_layers (int): Number of hidden layers.
        neurons_per_layer (int): Number of neurons per hidden layer.
        activation (str): Activation function (ReLU, Sigmoid, Tanh).
        dropout_rate (float): Dropout rate to prevent overfitting.
        optimizer (str): Optimizer to use (Adam, SGD, RMSprop).
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        model: Compiled ANN model.
    """
    
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation=activation.lower(), input_shape=(input_shape,)))
    model.add(Dropout(dropout_rate))
    
    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation.lower()))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation="sigmoid"))  # Binary classification output
    
    optimizers = {"Adam": Adam(learning_rate), "SGD": SGD(learning_rate), "RMSprop": RMSprop(learning_rate)}
    model.compile(loss="binary_crossentropy", optimizer=optimizers.get(optimizer, Adam(learning_rate)), metrics=["accuracy"])
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=50):
    """
    Trains the ANN model with given hyperparameters.
    
    Args:
        model: Compiled ANN model.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
    
    Returns:
        history: Training history containing loss and accuracy metrics.
    """
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs, verbose=1)
    
    return history
