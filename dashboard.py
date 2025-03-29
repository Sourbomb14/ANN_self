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
st.set_page_config(page_title="ANN Model Dashboard - Conversion Prediction", layout="wide")

# --- 1. Data Loading and Preprocessing ---

# Load Dataset and Model
DATASET_FILE_ID = "1OPmMFUQmeZuaiYb0FQhwOMZfEbVrWKEK"
MODEL_FILE_ID = "1NNxt6hnkAxUO8aI2sNCzPut0Nbmp8H_T"
MODEL_PATH = "conversion_model.h5"

if not os.path.exists("data.csv"):
    gdown.download(f"https://drive.google.com/uc?id={DATASET_FILE_ID}", "data.csv", quiet=False)

if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

# App Title and Logo
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

# --- Model Training ---
if st.button("🚀 Train Model"):
    with st.spinner("Training model... ⏳"):
        df = pd.read_csv("data.csv")
        df = df.sample(n=min(50000, len(df)), random_state=552627)

        # Feature Selection
        features = ['Age', 'Gender', 'Income', 'Purchases', 'Clicks', 'Spent']
        target = 'Converted'

        # Convert to Numeric & Handle NaNs
        df[features + [target]] = df[features + [target]].apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)

        # Encode Categorical Data
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

        # Build and Train Model
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(shape=(X_train.shape[1],)))
        for _ in range(dense_layers):
            model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation_function))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_split=0.2, class_weight=class_weight_dict, verbose=0)

    st.success("🎉 Model training complete!")

    # Evaluation Metrics
    st.subheader("📊 Model Performance")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.metric(label="Test Accuracy", value=f"{accuracy:.4f}")
    st.metric(label="Test Loss", value=f"{loss:.4f}")

    # Feature Importance using SHAP
    st.subheader("🔍 Feature Importance")
    explainer = shap.Explainer(model, X_train[:50])  # Reduced sample for performance
    shap_values = explainer(X_test[:50])  # Reduced sample for efficiency
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:50], show=False)
    st.pyplot(fig)

# Follow Me on GitHub Button
st.sidebar.markdown("### Follow Me on GitHub")
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Follow-blue?logo=github)](https://github.com/Rushil-K)")
