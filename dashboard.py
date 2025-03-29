import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import shap
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
from PIL import Image

# Set Streamlit Page Configuration
st.set_page_config(page_title="ANN Conversion Prediction", layout="wide")

# --- 1. Data Loading and Preprocessing ---

# 📥 Load Dataset
DATASET_FILE_ID = "1OPmMFUQmeZuaiYb0FQhwOMZfEbVrWKEK"
MODEL_FILE_ID = "1NNxt6hnkAxUO8aI2sNCzPut0Nbmp8H_T"  # Add this line to define the model file ID
MODEL_PATH = "conversion_model.h5" # Define a path for the model

if not os.path.exists("data.csv"):
    gdown.download(f"https://drive.google.com/uc?id={DATASET_FILE_ID}", "data.csv", quiet=False)

if not os.path.exists(MODEL_PATH): # Download the model if it doesn't exist
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)


# 🎨 Updated Custom CSS for Enhanced UI - Coral and Teal Theme
st.markdown(
    """
    <style>
        /* Updated color scheme */
        .title {
            color: #FF7F50; /* Coral */
            text-align: center;
            margin-bottom: 2rem;
        }
        .sidebar-header {
            color: #008080; /* Teal */
            font-size: 1.5em;
            margin-bottom: 1rem;
        }
        .metric-label {
            font-size: 1.2em;
            color: #2C3E50; /* Dark gray */
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #F0B27A; /* Light Coral */
        }
        .stButton > button {
            background-color: #008080; /* Teal */
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            width: 100%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .stButton > button:hover {
            background-color: #006060; /* Darker Teal */
            box-shadow: 0 3px 7px rgba(0,0,0,0.3);
        }
        .stSlider > div > div > div > div {
            background-color: #008080; /* Teal */
        }
        .stSelectbox > div > div {
            border-color: #008080; /* Teal */
        }
        .stSelectbox > div > div:focus-within {
            border-color: #006060; /* Darker Teal */
            box-shadow: 0 0 0 3px rgba(0, 128, 128, 0.3); /* Teal focus shadow */
        }
        .dataframe {
            border: 1px solid #E0E0E0;
            border-radius: 5px;
            padding: 10px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow-x: auto;
            margin-bottom: 20px;
        }
        .stPlotlyChart {
            margin-bottom: 20px;
            border: 1px solid #E0E0E0;
            border-radius: 5px;
            padding: 10px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 🖼️ App Title and Logo
st.title("📊 ANN Model Dashboard - Conversion Prediction")

# Sidebar for Hyperparameter Tuning
st.sidebar.header("🔧 Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", 5, 12, 10, 1)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "elu"], index=0)
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop", "adamax"], index=0)
dense_layers = st.sidebar.selectbox("Dense Layers", [2, 3, 4, 5], index=1)
neurons_per_layer = st.sidebar.selectbox("Neurons per Layer", [32, 64, 128, 256], index=2)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.05)

# Select Optimizer
optimizers = {
    "adam": Adam(learning_rate=learning_rate),
    "sgd": SGD(learning_rate=learning_rate),
    "rmsprop": RMSprop(learning_rate=learning_rate),
    "adamax": Adamax(learning_rate=learning_rate)
}
optimizer = optimizers[optimizer_choice]


# --- 2. Model Building ---
def build_model(input_shape, optimizer, activation, dense_layers, neurons, dropout_rate):
    """
    Builds a TensorFlow Keras Sequential model.

    Args:
        input_shape (tuple): Shape of the input data.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer to use.
        activation (str): Activation function for the dense layers.
        dense_layers (int): Number of dense layers.
        neurons (int): Number of neurons per dense layer.
        dropout_rate (float): Dropout rate.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=(input_shape,)))
    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --- 4. Model Training and Evaluation ---
if st.button("🚀 Train Model"):
    with st.spinner("Training model... ⏳"):
        # Load the entire dataset
        df = pd.read_csv("data.csv")

        # Randomly sample 50,000 rows
        df = df.sample(n=50000, random_state=552627) # sampling 50000 rows

        # 🎯 Feature Selection
        features = ['Age', 'Gender', 'Income', 'Purchases', 'Clicks', 'Spent']
        target = 'Converted'

        # 🔄 Encode Categorical Features
        encoder = OrdinalEncoder()
        df[['Gender']] = encoder.fit_transform(df[['Gender']])

        # Handle Class Imbalance with SMOTE
        X = df[features]
        y = df[target]
        smote = SMOTE(random_state=552627)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

        # Standardize Data
        scaler = StandardScaler()
        X_resampled[X.columns] = scaler.fit_transform(X_resampled)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=552627)

        # Compute Class Weights
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        # Build and train the model
        model = build_model(X_train.shape[1], optimizer, activation_function, dense_layers, neurons_per_layer, dropout_rate)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2, class_weight=class_weight_dict, verbose=0)

    st.success("🎉 Model training complete!")

    # Model Performance Evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Display Metrics
    st.subheader("📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<p class='metric-label'>Test Accuracy</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{accuracy:.4f}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p class='metric-label'>Test Loss</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{loss:.4f}</p>", unsafe_allow_html=True)

    # Training Performance Plots
    st.subheader("📈 Training Performance")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(history.history['accuracy'], label="Train Accuracy", color="#FF7F50")  # Coral
    ax[0].plot(history.history['val_accuracy'], label="Validation Accuracy", color="#008080")  # Teal
    ax[0].set_title("Accuracy over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(history.history['loss'], label="Train Loss", color="#FF7F50") # Coral
    ax[1].plot(history.history['val_loss'], label="Validation Loss", color="#008080")  # Teal
    ax[1].set_title("Loss over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)
    st.pyplot(fig)

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["Not Converted", "Converted"], yticklabels=["Not Converted", "Converted"]) # changed cmap
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("📜 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    # ROC Curve and AUC
    st.subheader("📈 ROC Curve and AUC")
    y_pred_proba = model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(fpr, tpr, color="#FF7F50", lw=2, label=f"AUC = {roc_auc:.2f}") # Coral
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True)
    st.pyplot(fig)

    # Feature Importance using SHAP
    st.subheader("🔍 Feature Importance")
    explainer = shap.Explainer(model, X_train[:100])
    shap_values = explainer(X_test[:100])
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:100], show=False)
    st.pyplot(fig)

    # Feature Importance Stats
    st.subheader("📌 Feature Importance Stats")
    mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': mean_abs_shap_values})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    st.dataframe(importance_df)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# --- 5. Make Predictions ---
st.subheader("🔮 Predict Conversion for a New Customer")

# Input fields for the new customer's data
age = st.number_input("Age", min_value=18, max_value=80, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
income = st.number_input("Income", min_value=10000, max_value=100000, value=50000)
purchases = st.number_input("Number of Purchases", min_value=0, max_value=100, value=5)
clicks = st.number_input("Number of Clicks", min_value=0, max_value=1000, value=100)
spent = st.number_input("Amount Spent", min_value=0.0, max_value=10000.0, value=1000.0)

# Encode gender
gender_encoded = 0 if gender == "Male" else 1

# Prepare input data for prediction
new_customer_data = np.array([[age, gender_encoded, income, purchases, clicks, spent]])

# Standardize the input data using the same scaler used during training
scaler = StandardScaler()
#dummy_data to avoid error
dummy_train = np.array([[18, 0, 10000, 0, 0, 1000.0], [80, 1, 100000, 100, 1000, 10000.0]])
scaler.fit(dummy_train)
new_customer_scaled = scaler.transform(new_customer_data)


# Predict Conversion
if st.button("Predict Conversion"):
    prediction = model.predict(new_customer_scaled)
    conversion_probability = prediction[0][0]
    st.write(f"Conversion Probability: {conversion_probability:.4f}")

# Display the first 5 rows of the dataframe
st.subheader("Sample Data")
df = pd.read_csv("data.csv") #load the data
st.dataframe(df.head())

# Display descriptive statistics for the numerical columns
st.subheader("Descriptive Statistics")
st.dataframe(df.describe())

#  create pair plots for the numerical features
st.subheader("Pair Plots")
fig = sns.pairplot(df[['Age', 'Income', 'Purchases', 'Clicks', 'Spent', 'Converted']], hue='Converted')
st.pyplot(fig)

# GitHub Follow Button
st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem;">
        <a href="https://github.com/Rushil-K" target="_blank" rel="noopener noreferrer">
            <button style="background-color: #008080; color: white; padding: 12px 24px; font-size: 18px; border: none; border-radius: 6px; cursor: pointer; transition: background-color 0.3s ease;">
                ⭐ Follow Me on GitHub
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
