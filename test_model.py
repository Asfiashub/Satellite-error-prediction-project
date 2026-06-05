# test_model.py

import tensorflow as tf

print("TensorFlow:", tf.__version__)

model = tf.keras.models.load_model(
    "models/GEO_satclockerror (m)_model.h5",
    compile=False
)

print("SUCCESS")