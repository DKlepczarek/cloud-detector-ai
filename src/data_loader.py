"""
data_loader.py
Handles data loading and augmentation pipeline using TensorFlow/Keras.
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    """
    Creates training and validation generators with data augmentation.
    """
    # Augmentacja: uczymy model, że obrócona chmura to nadal chmura
    train_datagen = ImageDataGenerator(
        rescale=1./255,         # Normalizacja pikseli 0-255 -> 0-1
        rotation_range=20,      # Lekki obrót
        zoom_range=0.2,         # Przybliżenie
        horizontal_flip=True,   # Lustrzane odbicie
        validation_split=0.2    # 20% danych na walidację
    )

    # Generator treningowy
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Generator walidacyjny (tylko skalowanie, bez obracania!)
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator