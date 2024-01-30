import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.applications import VGG16
import numpy as np
import shap
import os

# Configure TensorFlow to use dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Path to a single image for testing
image_path = '/home/darksst/Desktop/SHAP_Project/fruits-360-original-size/fruits-360-original-size/Test/apple_6/r0_259.jpg'  # Replace with the path to your image

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make a prediction
predictions = model.predict(img_array)

# Decode the prediction
decoded_predictions = decode_predictions(predictions, top=3)[0]
print("Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"   {i + 1}: {label} ({score:.2f})")

# Initialize the SHAP explainer
explainer = shap.DeepExplainer(model, img_array)

# Compute SHAP values
shap_values = explainer.shap_values(img_array)

# Visualize the SHAP values
shap.image_plot(shap_values, -img_array)

