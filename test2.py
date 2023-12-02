import shap
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array, load_img

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Load and preprocess an image
image = load_img('path_to_image.jpg', target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

# Create a SHAP DeepExplainer
explainer = shap.DeepExplainer(model, image)

# Calculate SHAP values
shap_values = explainer.shap_values(image)

# Plot the SHAP values
shap.image_plot(shap_values, -image)

