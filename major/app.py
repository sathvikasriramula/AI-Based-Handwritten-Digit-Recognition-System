```python
import os 
import numpy as np 
import streamlit as st 
import tensorflow as tf 
 
from PIL import Image, ImageOps 
 
 
st.set_page_config( 
    page_title="Handwritten Digit Recognition", 
    page_icon="🔢", 
    layout="wide" 
) 
 
 
st.title("🔢 AI-Based Handwritten Digit Recognition") 
 
st.write( 
    "Upload a handwritten digit and let the CNN model " 
    "identify it." 
) 
 
 
MODEL_PATH = "digit_cnn.keras" 
 
 
@st.cache_resource 
def load_model(): 
 
    return tf.keras.models.load_model( 
        MODEL_PATH 
    ) 
 
 
if not os.path.exists(MODEL_PATH): 
 
    st.error( 
        "❌ Model file 'digit_cnn.keras' was not found." 
    ) 
 
    st.info( 
        "Run 'python train_model.py' first." 
    ) 
 
    st.stop() 
 
 
model = load_model() 
 
 
st.sidebar.title("📌 Project Information") 
 
st.sidebar.write( 
    """ 
    **Dataset:** MNIST 
 
    **Training Images:** 60,000 
 
    **Testing Images:** 10,000 
 
    **Image Size:** 28 × 28 
 
    **Classes:** 0–9 
 
    **Model:** CNN 
 
    **Framework:** TensorFlow/Keras 
    """ 
) 
 
 
st.header("📤 Upload Digit Image") 
 
uploaded_file = st.file_uploader( 
    "Select a PNG, JPG or JPEG image", 
    type=["png", "jpg", "jpeg"] 
) 
 
 
def preprocess_image(image): 
 
    image = image.convert("L") 
 
    image = ImageOps.invert(image) 
 
    image = image.resize( 
        (28, 28) 
    ) 
 
    image_array = np.array(image) 
 
    image_array = ( 
        image_array.astype("float32") / 255.0 
    ) 
 
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
 
    image = Image.open( 
        uploaded_file 
    ) 
 
 
    st.markdown("---") 
 
    col1, col2 = st.columns(2) 
 
 
    with col1: 
 
        st.subheader( 
            "🖼️ Original Image" 
        ) 
 
        st.image( 
            image, 
            width=300 
        ) 
 
 
    processed_image = preprocess_image( 
        image 
    ) 
 
 
    with col2: 
 
        st.subheader( 
            "⚙️ Processed Image" 
        ) 
 
        st.image( 
            processed_image[0].squeeze(), 
            width=300 
        ) 
 
 
    prediction = model.predict( 
        processed_image, 
        verbose=0 
    ) 
 
 
    predicted_digit = int( 
        np.argmax( 
            prediction[0] 
        ) 
    ) 
 
 
    confidence = float( 
        np.max( 
            prediction[0] 
        ) 
    ) * 100 
 
 
    st.markdown("---") 
 
    st.header( 
        "🎯 Prediction Result" 
    ) 
 
 
    result_col1, result_col2, result_col3 = st.columns( 
        [1, 2, 1] 
    ) 
 
 
    with result_col2: 
 
        st.success( 
            "Prediction completed successfully!" 
        ) 
 
        st.metric( 
            label="🔢 Predicted Digit", 
            value=str(predicted_digit) 
        ) 
 
        st.metric( 
            label="📊 Confidence", 
            value=f"{confidence:.2f}%" 
        ) 
 
 
    st.write(""
```
