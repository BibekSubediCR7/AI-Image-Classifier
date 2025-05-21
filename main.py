import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image


def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")

    st.markdown(
        """
        <h1 style='text-align: center; color: #4B8BBE;'>🖼️ AI Image Classifier</h1>
        <p style='text-align: center; font-size: 18px;'>
            Upload an image and let AI identify what's in it using a pre-trained MobileNetV2 model.
        </p>
        <hr style='margin-top: 0;'>
        """,
        unsafe_allow_html=True
    )

    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with col2:
            if st.button("🔍 Classify Image"):
                with st.spinner("Analyzing Image..."):
                    image = Image.open(uploaded_file)
                    predictions = classify_image(model, image)

                    if predictions:
                        st.success("✅ Top Predictions")
                        for _, label, score in predictions: 
                            st.markdown(f"<p style='font-size:18px;'>🧠 <b>{label}</b>: {score:.2%}</p>", unsafe_allow_html=True)

    st.markdown(
        """
        <hr>
        <p style='text-align: center; font-size: 14px; color: gray;'>
            © 2025 | Made with ❤️ by <b>Bibek Subedi</b><br>
            Powered by <a href='https://www.tensorflow.org/' target='_blank'>TensorFlow</a> &nbsp;|&nbsp; 
            UI via <a href='https://streamlit.io/' target='_blank'>Streamlit</a>
        </p>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
