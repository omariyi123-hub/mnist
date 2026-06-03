
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================
# ✅ PAGE CONFIG
# =========================
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🤖 AI Digit Recognizer")
st.write("Upload a digit image (0–9)")

# =========================
# ✅ LOAD MODEL (SAFE)
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.keras")

model = load_model()

# =========================
# ✅ PREPROCESS FUNCTION
# =========================
def preprocess_image(img):
    img = img.convert("L")              # grayscale
    img = img.resize((28, 28))          # resize
    img = np.array(img)

    img = 255 - img                     # invert (important)
    img = img / 255.0                   # normalize

    img = img.reshape(1, 28, 28, 1)     # reshape
    return img

# =========================
# ✅ UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    if st.button("🚀 Predict"):
        img = preprocess_image(image)
        prediction = model.predict(img)

        st.success(f"✅ Predicted Digit: {np.argmax(prediction)}")
