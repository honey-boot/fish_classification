import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Fish Classifier 🐟", layout="centered")

# Load your best model
model = load_model("data/models/best_model.h5")

# Class labels
class_labels = sorted(os.listdir("data/train"))

# UI
st.title("🐠 Fish Image Classifier")
st.write("Upload a fish image and get the predicted species along with confidence scores.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload fish image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence_score = prediction[0][predicted_index] * 100

    # Show prediction
    st.markdown(f"### ✅ Predicted: **{predicted_label}**")
    st.markdown(f"**Confidence:** {confidence_score:.2f}%")

    # Top 3
    st.subheader("🔍 Top 3 Predictions:")
    top_indices = prediction[0].argsort()[-3:][::-1]
    for i in top_indices:
        st.write(f"{class_labels[i]}: {prediction[0][i]*100:.2f}%")

    # Chart
    st.subheader("📊 Class Probability Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(class_labels, prediction[0] * 100, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Confidence (%)")
    plt.title("Confidence Scores for Each Class")
    st.pyplot(fig)
else:
    st.info("👈 Upload a fish image to begin classification.")