import tensorflow as tf
from utils import *

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same',
                               input_shape=(IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_LABELS, activation='softmax')
    ])
    return model

def save_model(model, name):
    model.save(MODEL_DIR + name)

def load_model(name):
    return tf.keras.models.load_model(MODEL_DIR + name)