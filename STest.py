import json
import shap
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image
import numpy as np

# Load pre-trained ResNet50 model
model = ResNet50(weights="imagenet")

# Load a sample dataset from SHAP
# This loads the first 50 images from the ImageNet dataset
X, y = shap.datasets.imagenet50()

# Load ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with open(shap.datasets.cache(url)) as file:
    class_names = [v[1] for v in json.load(file).values()]

# Define a custom prediction function that preprocesses input images
def f(x):
    x = preprocess_input(x.copy())
    return model.predict(x)

# Define a masker that is used to mask out parts of the input image
masker = shap.maskers.Image("inpaint_telea", X[0].shape)

# Create a SHAP explainer object
explainer = shap.Explainer(f, masker, output_names=class_names)

# Generate SHAP values for a subset of the dataset
shap_values = explainer(X[:3], max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])

# Visualization of the SHAP values
shap.image_plot(shap_values, -X[:3])

