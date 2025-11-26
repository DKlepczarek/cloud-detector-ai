import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from src.utils import get_advice

# Class list
CLASS_NAMES = [
    'Altocumulus',   # Ac
    'Altostratus',   # As
    'Cumulonimbus',  # Cb 
    'Cirrocumulus',  # Cc
    'Cirrus',        # Ci
    'Cirrostratus',  # Cs
    'Contrail',      # Ct 
    'Cumulus',       # Cu
    'Nimbostratus',  # Ns
    'Stratocumulus', # Sc
    'Stratus'        # St
]

def load_and_prep_image(image_path, img_size=(224, 224)):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array

def main(image_path, model_path='models/cloud_model_v1.h5'):
    
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except OSError:
        print("Error: Model not found. Train the model first!")
        return

    # Prediction
    img = load_and_prep_image(image_path)
    prediction = model.predict(img)
    
    # Extracting result
    score = tf.nn.softmax(prediction[0])
    class_idx = np.argmax(prediction)
    class_name = CLASS_NAMES[class_idx]
    confidence = np.max(prediction)

    # Business logic (Skipper's advice)
    advice = get_advice(class_name)

    print("\n" + "="*40)
    print(f"Cloud Type: {class_name}")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 40)
    print(f"SAILING ADVICE: {advice['message']}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image')
    args = parser.parse_args()
    main(args.image_path)