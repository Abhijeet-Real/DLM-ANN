import pandas as pd
from DataPreProcessor import DataPreprocessor
from model import build_ann, train_model  # Assuming model.py contains ANN functions

def load_and_train_model(DATASET_FILE: str, hyperparams: dict):
    """
    Loads the dataset, preprocesses it, and trains an ANN model using given hyperparameters.
    
    Args:
        DATASET_FILE (str): Path to the dataset file.
        hyperparams (dict): Dictionary containing ANN hyperparameters.
    
    Returns:
        model: Trained ANN model.
        history: Training history containing loss and accuracy metrics.
    """
    
    # Load Dataset from local environment
    df = pd.read_csv(DATASET_FILE)
    
    # Initialize Data Preprocessor
    preprocessor = DataPreprocessor(
        dataframe=df,
        target_variable="num",  # Assuming 'num' is the target column
        train_test_split_percentage=hyperparams.get("train_test_split", 80)
    )
    
    # Perform preprocessing
    X_train, X_test, y_train, y_test = preprocessor.pre_process()
    
    # Build ANN Model
    model = build_ann(
        input_shape=X_train.shape[1],
        num_layers=hyperparams.get("num_layers", 2),
        neurons_per_layer=hyperparams.get("neurons_per_layer", 32),
        activation=hyperparams.get("activation", "ReLU"),
        dropout_rate=hyperparams.get("dropout_rate", 0.2),
        optimizer=hyperparams.get("optimizer", "Adam"),
        learning_rate=hyperparams.get("learning_rate", 0.001)
    )
    
    # Train Model
    history = train_model(
        model,
        X_train, y_train,
        X_test, y_test,
        batch_size=hyperparams.get("batch_size", 32),
        epochs=hyperparams.get("epochs", 50)
    )
    
    return model, history