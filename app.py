
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas

# =========================
# ✅ PAGE CONFIG
# =========================
st.set_page_config(page_title="Digit Recognizer", layout="centered")

st.title("🤖 AI Digit Recognizer")
st.write("Upload OR draw a digit (0–9)")

# =========================
# ✅ LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

model = load_model()

# =========================
# ✅ PERFECT PREPROCESSING (VERY IMPORTANT)
# =========================
def preprocess_image(img):
    img = img.convert("L")
    img = np.array(img)

    # invert colors if needed
    if np.mean(img) > 127:
        img = 255 - img

    # threshold
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # find bounding box
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return None

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    img = img[x_min:x_max + 1, y_min:y_max + 1]

    # resize keeping aspect ratio → 20px
    h, w = img.shape

    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    img = cv2.resize(img, (new_w, new_h))

    # pad to 28x28
    pad_h = (28 - new_h) // 2
    pad_w = (28 - new_w) // 2

    img = np.pad(
        img,
        ((pad_h, 28 - new_h - pad_h),
         (pad_w, 28 - new_w - pad_w)),
        mode='constant'
    )

    # normalize
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img

# =========================
# ✅ TABS (UPLOAD + DRAW)
# =========================
tab1, tab2 = st.tabs(["📤 Upload Image", "✍️ Draw Digit"])

uploaded_image = None
canvas_image = None

# =========================
# 📤 UPLOAD TAB
# =========================
with tab1:
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, width=200)


# =========================
# ✍️ DRAW TAB (FIXED ONLY THIS PART)
# =========================

with tab2:
    st.subheader("✍️ Draw Digit (works properly)")

    canvas_result = st_canvas(
        stroke_width=15,
        stroke_color="#FFFFFF",         # ✅ force white
        background_color="#000000",     # ✅ force black
        height=300,
        width=300,
        drawing_mode="freedraw",
        update_streamlit=True,          # ✅ REQUIRED for interaction
        key="canvas_fixed",
    )

    # ✅ SHOW RAW DRAWING STATUS
    if canvas_result.json_data is not None:
        st.write("✅ Drawing working")

    # ✅ Extract image ONLY when something drawn
    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:, :, 0].astype("uint8")

        canvas_image = Image.fromarray(img, mode="L")

        st.image(canvas_image, width=150)


# =========================
# ✅ PREDICT BUTTON
# =========================
st.markdown("---")

if st.button("🚀 Predict"):

    image = None

    if uploaded_image is not None:
        image = uploaded_image
    elif canvas_image is not None:
        image = canvas_image

    if image is None:
        st.warning("Please upload or draw a digit")
        st.stop()

    processed = preprocess_image(image)

    if processed is None:
        st.error("Could not detect digit")
        st.stop()

    # predict
    pred = model.predict(processed)
    digit = np.argmax(pred)

    st.success(f"✅ Predicted Digit: {digit}")

    st.subheader("📊 Confidence")
    st.bar_chart(pred[0])

