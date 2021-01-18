import tensorflow as tf

MODEL_PATH = './cancer.keras'

def load_keras_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model
