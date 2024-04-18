import json
import tensorflow as tf
import numpy as np
import shap
from keras.applications.vgg16 import VGG16, preprocess_input

# Load pre-trained VGG16 model
model = VGG16(weights="imagenet", include_top=True)

# Load a subset of ImageNet data for explanations
X, y = shap.datasets.imagenet50()
to_explain = X[[39, 41]]

# Load ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)

# Preprocess the data
X_processed = preprocess_input(X.copy())

# Create SHAP GradientExplainer using the model
e = shap.GradientExplainer(
    model,
    X_processed
)

# Compute SHAP values
shap_values, indexes = e.shap_values(to_explain, ranked_outputs=2)

# Get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# Plot the explanations
shap.image_plot(shap_values, to_explain, index_names)

