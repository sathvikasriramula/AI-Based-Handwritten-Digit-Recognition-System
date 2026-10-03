```python
import os 
 
import numpy as np 
import streamlit as st 
import tensorflow as tf 
 
from PIL import Image, ImageOps 
 
 
st.set_page_config( 
    page_title="AI Handwritten Digit Recognition", 
    page_icon="🔢", 
    layout="wide" 
) 
 
 
st.markdown( 
    """ 
    <style> 
 
    .main { 
        background-color: #f5f7fb; 
    } 
 
    .title { 
        text-align: center; 
        font-size: 40px; 
        font-weight: bold; 
        margin-bottom: 5px; 
    } 
 
    .subtitle { 
        text-align: center; 
        font-size: 18px; 
        margin-bottom: 30px; 
    } 
 
    </style> 
    """, 
    unsafe_allow_html=True 
) 
 
 
st.markdown( 
    '<div class="title">🔢 AI-Based Handwritten Digit Recognition</div>', 
    unsafe_allow_html=True 
) 
 
st.markdown( 
    '<div class="subtitle">' 
    'Deep Learning based handwritten digit classification using CNN' 
    '</div>', 
    unsafe_allow_html=True 
) 
 
 
MODEL_PATH = "digit_cnn.keras" 
 
 
@st.cache_resource 
def load_model(): 
 
    if not os.path.exists(MODEL_PATH): 
        return None 
 
    return tf.keras.models.load_model(MODEL_PATH) 
 
 
model = load_model() 
 
 
if model is None: 
 
    st.error( 
        "❌ CNN model not found." 
    ) 
 
    st.info( 
        "Please run 'python train_model.py' first." 
    ) 
 
    st.stop() 
 
 
st.sidebar.title("📌 Project Information") 
 
st.sidebar.info( 
    """ 
    **Dataset:** MNIST 
 
    **Training Images:** 60,000 
 
    **Testing Images:** 10,000 
 
    **Image Size:** 28 × 28 
 
    **Classes:** 0 – 9 
 
    **Model:** CNN 
 
    **Framework:** TensorFlow / Keras 
    """ 
) 
 
 
st.sidebar.markdown("---") 
 
st.sidebar.subheader("🧠 CNN Architecture") 
 
st.sidebar.write( 
    """ 
    Input Image 
    ↓ 
    Conv2D 
    ↓ 
    MaxPooling 
    ↓ 
    Conv2D 
    ↓ 
    MaxPooling 
    ↓ 
    Conv2D 
    ↓ 
    Flatten 
    ↓ 
    Dense 
    ↓ 
    Dropout 
    ↓ 
    Softmax 
    """ 
) 
 
 
st.header("📤 Upload Handwritten Digit") 
 
uploaded_file = st.file_uploader( 
    "Choose an image containing a handwritten digit", 
    type=["png", "jpg", "jpeg"] 
) 
 
 
def preprocess_image(image): 
 
    image = image.convert("L") 
 
    image = ImageOps.invert(image) 
 
    image = image.resize((28, 28)) 
 
    image_array = np.array(image) 
 
    image_array = image_array.astype( 
        "float32" 
    ) / 255.0 
 
    image_array = np.expand_dims( 
        image_array, 
        axis=-1 
    ) 
 
    image_array = np.expand_dims( 
        image_array, 
        axis=0 
    ) 
 
    return image_array 
 
 
if uploaded_file is not None: 
 
    image = Image.open(uploaded_file) 
 
 
    st.markdown("---") 
 
    col1, col2 = st.columns(2) 
 
 
    with col1: 
 
        st.subheader("🖼️ Original Image") 
 
        st.image( 
            image, 
            width=300 
        ) 
 
 
    processed_image = preprocess_image( 
        image 
    ) 
 
 
    with col2: 
 
        st.subheader("⚙️ Processed Image") 
 
        st.image( 
            processed_image[0].squeeze(), 
            width=300 
        ) 
 
 
    prediction = model.predict( 
        processed_image, 
        verbose=0 
    ) 
 
 
    predicted_digit = int( 
        np.argmax(prediction[0]) 
    ) 
 
 
    confidence = float( 
        np.max(prediction[0]) 
    ) * 100 
 
 
    st.markdown("---") 
 
    st.header("🎯 Prediction Result") 
 
 
    result_col1, result_col2, result_col3 = st.columns( 
        [1, 2, 1] 
    ) 
 
 
    with result_col2: 
 
        st.metric( 
            label="Predicted Digit", 
            value=str(predicted_digit) 
        ) 
 
        st.metric( 
            label="Confidence", 
            value=f"{confidence:.2f}%" 
        ) 
 
 
    st.success( 
        f"The model predicts the digit as {predicted_digit} " 
        f"with {confidence:.2f}% confidence." 
    ) 
 
 
    st.markdown("---") 
 
    st.header( 
        "📊 Prediction Probability for Each Digit" 
    ) 
 
 
    probabilities = prediction[0] 
 
 
    probability_data = {} 
 
    for digit in range(10): 
 
        probability_data[ 
            str(digit) 
        ] = float( 
            probabilities[digit] 
        ) 
 
 
    st.bar_chart( 
        probability_data 
    ) 
 
 
    st.subheader( 
        "Detailed Probabilities" 
    ) 
 
 
    for digit in range(10): 
 
        probability = float( 
            probabilities[digit] 
        ) 
 
        percentage = probability * 100 
 
 
        st.write( 
            f"Digit **{digit}**: " 
            f"{percentage:.2f}%" 
        ) 
 
        st.progress( 
            probability 
        ) 
 
 
else: 
 
    st.markdown("---") 
 
    st.info( 
        "👆 Upload a handwritten digit image above " 
        "to start prediction." 
    ) 
 
 
st.markdown("---") 
 
st.header("🧠 About the Project") 
 
st.write( 
    """ 
    This project uses a Convolutional Neural Network (CNN) 
    to recognize handwritten digits from 0 to 9. 
 
    The model is trained using the MNIST dataset containing 
    60,000 training images and 10,000 testing images. 
 
    Each input image is converted to grayscale, resized to 
    28 × 28 pixels, normalized and passed to the CNN model. 
 
    The model then predicts the digit and provides a 
    confidence score. 
    """ 
) 
 
 
st.markdown("---") 
 
st.caption( 
    "AI-Based Handwritten Digit Recognition System | " 
    "Python + TensorFlow + Keras + Streamlit" 
)

