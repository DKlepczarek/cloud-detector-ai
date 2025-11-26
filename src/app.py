import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# === PAGE CONFIGURATION ===
st.set_page_config(page_title="Cloud Atlas AI", page_icon="⛵")

# === TITLE AND DESCRIPTION ===
st.title("☁️ Cloud Detector AI")
st.write("### Intelligent Assistant for Offshore Sailors")
st.write("Upload a photo of the sky to detect cloud types and get sailing safety advice.")

# === MODEL LOADING (Cached for performance) ===
# We use st.cache_resource to load the model only once, not on every refresh.
@st.cache_resource
def load_model():
    # Path to the fine-tuned model
    model_path = 'models/cloud_model_best.h5'
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except OSError:
        st.error(f"Error: Model not found at {model_path}. Did you run the training?")
        return None

with st.spinner('Loading AI Model...'):
    model = load_model()

# === BUSINESS LOGIC & DOMAIN KNOWLEDGE ===
CLASS_NAMES = [
    'Altocumulus', 'Altostratus', 'Cumulonimbus', 'Cirrocumulus', 'Cirrus',
    'Cirrostratus', 'Contrail', 'Cumulus', 'Nimbostratus', 'Stratocumulus', 'Stratus'
]

def get_sailing_advice(cloud_type, is_storm_risk=False):
    """
    Returns specific sailing advice based on the detected cloud type.
    """
    if is_storm_risk:
        return " **!!!DANGER!!!** Storm imminent! Reef the sails immediately and close hatches.", "error"
    
    advice_dict = {
        "Cumulonimbus": (" **!!!DANGER!!!** Storm imminent! Reef the sails immediately.", "error"),
        "Cumulus": (" **SAFE:** Fair weather. Good conditions for full sails.", "success"),
        "Cirrus": (" **INFO:** Weather change approaching within 24h. Monitor barometer.", "info"),
        "Stratus": (" **STABLE:** Overcast, possible drizzle. Visibility might be reduced.", "warning"),
        "Nimbostratus": (" **RAIN:** Continuous rain likely. Prepare foul weather gear.", "warning"),
        "Stratocumulus": (" **STABLE:** No immediate threat, but wind may be gusty.", "info"),
    }
    
    # Default fallback advice
    return advice_dict.get(cloud_type, ("⚓ **Proceed with caution.** Monitor wind changes.", "info"))

# === USER INTERFACE ===

uploaded_file = st.file_uploader(
    "Choose a cloud image...", 
    type=["jpg", "png", "jpeg", "webp", "tiff", "bmp"]
)

if uploaded_file is not None and model is not None:
    
    # 1. Display and Process Image (UNIVERSAL CONVERTER)
    try:
        image_raw = Image.open(uploaded_file)
        image = image_raw.convert("RGB")
        st.image(image, caption='Uploaded Image (Converted to RGB)', use_container_width=True)
        
        # Resize image to model input size
        target_size = (224, 224)
        image_resized = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
        
        # Prepare data for the model
        img_array = np.array(image_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype(np.float32) / 255.0
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.stop()

    # 2. Run Inference
    if st.button('Analyze Sky Image'):
        prediction = model.predict(img_array)
        scores = prediction[0]
        
        # --- SAFETY LOGIC ---
        try:
            cb_index = CLASS_NAMES.index("Cumulonimbus")
            storm_prob = scores[cb_index]
        except ValueError:
            storm_prob = 0.0
        
        predicted_class_idx = np.argmax(scores)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = np.max(scores)

        st.divider()
        
        # --- RESULTS ---
        if storm_prob > 0.25: 
            st.error(f"**SAFETY OVERRIDE TRIGGERED**")
            st.write(f"Storm Probability: **{storm_prob:.1%}** (Threshold: 25%)")
            st.write("System detected storm features despite lower confidence.")
            
            final_class = "Cumulonimbus"
            advice, status = get_sailing_advice(final_class, is_storm_risk=True)
        else:
            st.success(f"Detection: **{predicted_class}**")
            st.write(f"Confidence: **{confidence:.1%}**")
            advice, status = get_sailing_advice(predicted_class)

        if status == "error": st.error(advice)
        elif status == "success": st.success(advice)
        elif status == "warning": st.warning(advice)
        else: st.info(advice)

        # 3. Visualization
        st.write("---")
        st.write("### Detailed Probabilities")
        chart_data = dict(zip(CLASS_NAMES, scores))
        st.bar_chart(chart_data)