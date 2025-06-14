import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Satellite Fire Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f50;
        margin-bottom: 15px;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 5px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .fire-detected {
        background-color: #ffcccc;
        border: 2px solid #ff4b4b;
        color: #d9534f;
    }
    .no-fire {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #28a745;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Satellite Fire Detection System</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2208/2208823.png", width=100)
    st.header("About")
    st.markdown("""
    This application uses a CNN model trained on satellite imagery to detect the presence of wildfires.
    
    The model was trained on the 'wildfire-prediction-dataset' from Kaggle with:
    - 30,250 training images
    - 6,300 validation images
    - 6,300 test images
    
    Upload your satellite image to check for wildfires!
    """)
    
    st.header("Dataset Credit")
    st.markdown("Dataset: [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)")

# Load model function
@st.cache_resource
def load_fire_model():
    """Load the pre-trained model"""
    try:
        model = load_model('fire_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess image function
def preprocess_image(image):
    """Preprocess the image for model prediction"""
    # Resize to match model input size
    img = image.resize((64, 64))
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict_fire(image, model):
    """Make prediction using the model"""
    try:
        # Preprocess image
        processed_img = preprocess_image(image)
        # Make prediction
        prediction = model.predict(processed_img)[0][0]
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Main content area
st.markdown("<h2 class='sub-header'>Upload Satellite Image</h2>", unsafe_allow_html=True)
st.write("Upload a satellite image to detect if it contains wildfire.")

# File uploader
uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

# Display and process the uploaded image
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Add predict button
    if st.button("Detect Fire"):
        with st.spinner("Analyzing image..."):
            # Load model
            model = load_fire_model()
            
            if model:
                # Make prediction
                prediction_score = predict_fire(image, model)
                
                if prediction_score is not None:
                    # Display prediction result
                    if prediction_score > 0.5:
                        st.markdown(f"<div class='result-box fire-detected'>WILDFIRE DETECTED (Confidence: {prediction_score:.2%})</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='result-box no-fire'>NO WILDFIRE DETECTED (Confidence: {(1-prediction_score):.2%})</div>", unsafe_allow_html=True)
                    
                    # Display prediction details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Prediction details:")
                        st.json({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "prediction_score": float(prediction_score),
                            "result": "Wildfire" if prediction_score > 0.5 else "No Wildfire"
                        })
                    
                    with col2:
                        # Display confidence meter
                        if prediction_score > 0.5:
                            st.write("Wildfire Confidence:")
                            st.progress(float(prediction_score))
                        else:
                            st.write("No Wildfire Confidence:")
                            st.progress(float(1-prediction_score))
else:
    # Display instructions when no file is uploaded
    st.info("Please upload a satellite image to begin fire detection")
    
    # Display example information
    st.markdown("""
    <div class='info-box'>
    <h3>What kind of images work best?</h3>
    <p>For best results, upload satellite or aerial imagery that shows:</p>
    <ul>
        <li>Clear landscape views from above</li>
        <li>Good resolution and lighting</li>
        <li>Minimal cloud cover</li>
        <li>Natural color images (not infrared or other special bands)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Information section
st.markdown("<h2 class='sub-header'>How It Works</h2>", unsafe_allow_html=True)
st.markdown("""
<div class='info-box'>
<h3>Model Information</h3>
<p>This application uses a Convolutional Neural Network (CNN) trained on satellite imagery to detect wildfires. The model analyzes patterns, colors, and textures in the image to identify fire signatures.</p>

<h3>Model Architecture</h3>
<p>The CNN model consists of:</p>
<ul>
    <li>Convolutional layers for feature extraction</li>
    <li>MaxPooling layers for dimensionality reduction</li>
    <li>Dropout layers to prevent overfitting</li>
    <li>Dense layers for classification</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Developed for wildfire detection and monitoring using satellite imagery. Built with TensorFlow and Streamlit.")
