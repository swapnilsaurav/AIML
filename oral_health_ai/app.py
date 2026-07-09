import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "oral_health_model.h5"
CLASS_FILE = "class_names.txt"

st.set_page_config(
    page_title="AI Oral Health Assistant",
    page_icon="🦷",
    layout="centered"
)

st.title("🦷 AI Oral Health Assistant")

st.write("""
Upload a clear photo of teeth or gums.  
The AI model will classify it into one of the trained categories.
""")

st.warning("""
This is an educational screening tool only. 
It is not a medical diagnosis. Please consult a dentist for confirmation.
""")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASS_FILE, "r") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
class_names = load_classes()

uploaded_file = st.file_uploader(
    "Upload teeth image",
    type=["jpg", "jpeg", "png"]
)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Image"):
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index] * 100

        predicted_class = class_names[predicted_index]

        st.subheader("Prediction Result")

        st.success(f"Detected Condition: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Class Probabilities")

        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {prediction[0][i] * 100:.2f}%")

        if predicted_class == "cavity":
            st.write("Possible cavity signs detected. Dental check-up is recommended.")
        elif predicted_class == "plaque":
            st.write("Plaque-like appearance detected. Improve brushing and dental cleaning.")
        elif predicted_class == "gum_problem":
            st.write("Possible gum issue detected. Please consult a dentist.")
        elif predicted_class == "healthy":
            st.write("The image appears closer to healthy teeth based on training data.")