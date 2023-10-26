import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="/Users/fawlys/Downloads/gettingWeights/app/src/main/assets/mobilenetv1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for i, layer in enumerate(interpreter.get_tensor_details()):
    print(f"Layer{i}: {layer['name']} - Shape: {layer['shape']} - Type: {layer['dtype']}")
    if 'quantization' in layer:
        print(f"Quantization parameters: {layer['quantization']}")
