import streamlit as st

def hyperparameter_filters():
    """
    Function to create a sidebar filter panel for hyperparameter tuning in Streamlit.
    Returns a dictionary with selected hyperparameters.
    """
    st.sidebar.header("🔧 Hyperparameter Tuning")
    
    # Model Architecture
    num_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)
    neurons_per_layer = st.sidebar.slider("Neurons per Layer", 8, 128, 32, step=8)
    activation = st.sidebar.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh"], index=0)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.05)
    
    # Training Parameters
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
    optimizer = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"], index=0)
    epochs = st.sidebar.slider("Epochs", 10, 200, 50, step=10)
    
    # Data Preprocessing
    scaling_method = st.sidebar.selectbox("Feature Scaling", ["Min-Max", "Standardization", "None"], index=0)
    train_test_split = st.sidebar.slider("Train-Test Split %", 60, 90, 80, step=5)
    
    # Collect all hyperparameters
    hyperparams = {
        "num_layers": num_layers,
        "neurons_per_layer": neurons_per_layer,
        "activation": activation,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "epochs": epochs,
        "scaling_method": scaling_method,
        "train_test_split": train_test_split / 100  # Convert to decimal
    }
    
    return hyperparams