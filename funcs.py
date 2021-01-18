import tensorflow as tf

LITE_MODEL = '/home/ubuntu/skincancerapp/cancer.tflite'
#LITE_MODEL = './cancer.tflite'

def get_prob(img):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=LITE_MODEL)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = img
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data[0][0]
