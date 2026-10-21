import streamlit as st
import numpy as np
from PIL import Image
import joblib
from pathlib import Path

# Model path
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "mnist_neural_network.pkl"
)

# Load trained neural network
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Application interface
st.title("Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (0–9).")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open image in grayscale
    image = Image.open(uploaded_file).convert("L")

    # Display original image
    st.image(
        image,
        caption="Uploaded Image",
        width=200
    )

    # Convert to NumPy array
    image_array = np.array(image)

    # Invert if background is white
    if np.mean(image_array) > 127:
        image_array = 255 - image_array

    # Remove faint background noise
    image_array[image_array < 30] = 0

    # Find digit boundaries
    coords = np.argwhere(image_array > 0)

    if coords.size == 0:
        st.warning("No digit detected in the image.")
        st.stop()

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop digit
    digit = image_array[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # Resize digit while preserving aspect ratio
    digit_image = Image.fromarray(digit)

    digit_image.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Create 28x28 MNIST-style canvas
    canvas = Image.new(
        "L",
        (28, 28),
        color=0
    )

    # Center digit
    x = (28 - digit_image.width) // 2
    y = (28 - digit_image.height) // 2

    canvas.paste(digit_image, (x, y))

    # Display processed image
    st.image(
        canvas.resize((280, 280)),
        caption="Processed MNIST Image",
        width=280
    )

    # Normalize pixel values
    processed_array = np.array(
        canvas,
        dtype=np.float32
    ) / 255.0

    # Flatten to 784 features
    processed_array = processed_array.reshape(1, 784)

    # Predict digit
    prediction = model.predict(processed_array)

    # Display result
    st.success(
        f"Predicted Digit: {prediction[0]}"
    )