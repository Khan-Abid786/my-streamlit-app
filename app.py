import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_wine

# Load Wine Dataset
wine = load_wine()
feature_names = wine.feature_names  # Names of the 13 features
target_names = wine.target_names    # Target names (e.g., class_0, class_1, class_2)

# Load your pre-trained model
model = pickle.load(open('/content/CompleteWineModelDataset', 'rb'))

# Title and description
st.title("Wine Classification Prediction")
st.write("This app predicts the class of wine based on 13 features of the Wine Dataset.")

# Input fields for the 13 features
st.sidebar.header("Input Features")
inputs = []
for feature in feature_names:
    value = st.sidebar.number_input(f"{feature.capitalize()}", value=0.0)
    inputs.append(value)

# Convert inputs to a numpy array
features = np.array([inputs])

# Predict button
if st.button("Predict"):
    prediction = model.predict(features)
    class_name = target_names[prediction[0]]
    st.success(f"The predicted wine class is: {class_name}")
