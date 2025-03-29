import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Set Streamlit Page Configuration
st.set_page_config(page_title="ANN Model Dashboard - Conversion Prediction", layout="wide")

# 📥 Load Dataset
DATASET_FILE_ID = "1OPmMFUQmeZuaiYb0FQhwOMZfEbVrWKEK"
MODEL_PATH = "conversion_model.h5"

if not os.path.exists("data.csv"):
    gdown.download(f"https://drive.google.com/uc?id={DATASET_FILE_ID}", "data.csv", quiet=False)

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading the model: {e}. Please ensure the model file is available.")
    model = None  # Prevent further errors if model fails to load

# Display the first 5 rows of the dataframe
st.subheader("Sample Data")
df = pd.read_csv("data.csv")
st.dataframe(df.head())

# Display descriptive statistics for the numerical columns
st.subheader("Descriptive Statistics")
st.dataframe(df.describe())

# Ensure numeric columns are properly formatted
df['Gender'] = OrdinalEncoder().fit_transform(df[['Gender']])
df = df[['Age', 'Gender', 'Income', 'Purchases', 'Clicks', 'Spent', 'Converted']]
df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
df = df.dropna()  # Drop any remaining NaN values

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Check GPU availability
if tf.config.list_physical_devices('GPU'):
    st.write("GPU is available. TensorFlow will use it if possible.")
else:
    st.write("No GPU available. TensorFlow will use the CPU.")
