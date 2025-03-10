import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from Filter import hyperparameter_filters
from DataPreProcessor import DataPreprocessor
from Connector import load_and_train_model

st.set_page_config(page_title="Heart Disease ANN Dashboard", layout="wide")

st.title("Heart Disease Prediction with ANN 🏥")

# Get hyperparameter selections from sidebar
hyperparams = hyperparameter_filters()

# Track training time
start_time = time.time()
model, history = load_and_train_model("heart_disease_uci.csv", hyperparams)
training_time = time.time() - start_time

# Display model summary
st.write("### Model Summary")
summary_string = []
model.summary(print_fn=lambda x: summary_string.append(x))
st.code("\n".join(summary_string), language="plaintext")

# Display training history
st.write("### Training History")

# Convert history to DataFrame
history_df = pd.DataFrame(history.history)
st.line_chart(history_df)

# Load test dataset for evaluation
st.write("### Model Evaluation")
preprocessor = DataPreprocessor(pd.read_csv("heart_disease_uci.csv"), target_variable="num")
X_train, X_test, y_train, y_test = preprocessor.pre_process()

# Track testing time
start_test_time = time.time()
y_pred = model.predict(X_test)
testing_time = time.time() - start_test_time

# Convert predictions to binary format
y_pred_binary = (y_pred > 0.5).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Plot Confusion Matrix
st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
st.pyplot(fig)

# Compute classification report
st.write("### Classification Report")
report = classification_report(y_test, y_pred_binary, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Compute ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
st.write("### ROC Curve")
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# Display training & testing time
st.write("### Training & Testing Time")
st.write(f"**Training Time:** {training_time:.2f} seconds")
st.write(f"**Testing Time:** {testing_time:.2f} seconds")
