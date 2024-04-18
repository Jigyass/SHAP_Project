import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import json
import numpy as np
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import os

# Load pre-trained model
model = VGG16(weights="imagenet", include_top=True)

# Load sample data
X, y = shap.datasets.imagenet50()
to_explain = X[[39, 41]]

# Load the ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)

# Function to map input to the output of the first layer
layer_of_interest = 1
def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
    return tf.compat.v1.keras.backend.get_session().run(model.layers[layer].input, feed_dict)

# Explain the first layer
e = shap.GradientExplainer(
    (model.layers[layer_of_interest].input, model.layers[-1].output),
    map2layer(preprocess_input(X.copy()), layer_of_interest),
)
shap_values, indexes = e.shap_values(map2layer(to_explain, layer_of_interest), ranked_outputs=2)

# Get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# Create a directory for SHAP outputs
output_dir = 'SHAP_Output'
os.makedirs(output_dir, exist_ok=True)

# Save the SHAP plots as images
for i, shap_array in enumerate(shap_values):
    class_index = indexes[0][i]
    class_name = class_names[str(class_index)][1]
    for j, image in enumerate(to_explain):
        plt.figure()
        shap.image_plot(shap_array, -np.array([image]), np.array([[class_name]]))
        plt.savefig(os.path.join(output_dir, f'Layer_{layer_of_interest}_Class_{class_name}_Image_{j}.png'))
        plt.close()
