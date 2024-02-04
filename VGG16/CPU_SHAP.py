import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
import numpy as np
import shap
# Ensure operations are run on CPU
with tf.device('/CPU:0'):
    print("Configured TensorFlow to use CPU.")

    # Path to a single image for testing
    image_path = '/home/darksst/Desktop/SHAP_Project/fruits-360-original-size/fruits-360-original-size/Test/apple_6/r0_259.jpg'  # Update this path
    print(f"Image path set to {image_path}.")

    # Load the pre-trained VGG16 model
    model = VGG16(weights='imagenet')
    print("Loaded VGG16 model with ImageNet weights.")

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    print("Loaded and resized the image.")
    img_array = image.img_to_array(img)
    print("Converted image to array.")
    img_array = np.expand_dims(img_array, axis=0)
    print("Expanded image array dimensions.")
    img_array = preprocess_input(img_array)
    print("Preprocessed the image array for VGG16.")

    # Make a prediction
    predictions = model.predict(img_array)
    print("Made predictions with the model.")
    # Decode the prediction
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    print("Decoded predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"   {i + 1}: {label} ({score:.2f})")

    # Function to map input to intermediate layer
    def map2layer(x, layer):
        # Preprocess and predict using the specific layer model
        intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[layer].output)
        intermediate_output = intermediate_layer_model(preprocess_input(x.copy()))
        return intermediate_output

    # Select layer 7 for SHAP explanation
    layer_to_explain = 7
    print(f"Preparing to explain layer {layer_to_explain}.")

    # Initialize SHAP GradientExplainer using an intermediate layer
    explainer = shap.GradientExplainer(
        (model.layers[layer_to_explain].input, model.layers[-1].output),
        map2layer(img_array, layer_to_explain).numpy(),
    )

    # Compute SHAP values
    shap_values = explainer.shap_values(map2layer(img_array, layer_to_explain).numpy(), ranked_outputs=2)

    # Visualize the SHAP values for the specific layer
    shap.image_plot(shap_values, -img_array)

