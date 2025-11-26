"""
model.py
Defines the CNN architecture using Transfer Learning (MobileNetV2).
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def build_model(num_classes, input_shape=(224, 224, 3)):
    """
    Builds a model based on pre-trained MobileNetV2.
    """
    # 1. Pobieramy "bazy" modelu (wytrenowane na ImageNet)
    # include_top=False oznacza, że odcinamy ostatnią warstwę klasyfikującą (1000 klas)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # 2. Zamrażamy wagi bazy (nie chcemy ich psuć podczas treningu)
    base_model.trainable = False

    # 3. Dodajemy własną "głowę" (Head) do klasyfikacji chmur
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Spłaszczamy wyniki
    x = Dense(128, activation='relu')(x) # Dodatkowa warstwa gęsta
    predictions = Dense(num_classes, activation='softmax')(x) # Warstwa wyjściowa

    # Składamy całość
    model = Model(inputs=base_model.input, outputs=predictions)

    # Kompilacja modelu
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model