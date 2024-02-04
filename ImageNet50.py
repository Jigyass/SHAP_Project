import json
import os
import numpy as np
import torch
from torchvision import models
import shap
import matplotlib.pyplot as plt

# Normalization function
def normalize(image):
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

# Load the model
model = models.vgg16(pretrained=True).eval()

# SHAP dataset
X, y = shap.datasets.imagenet50()
X /= 255
to_explain = X[[9, 41]]

# ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)

# SHAP Explainer
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
e = shap.GradientExplainer((model, model.features[7]), normalize(X))
shap_values, indexes = e.shap_values(normalize(to_explain), ranked_outputs=2, nsamples=200)

# Convert indexes to names
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# Adjust SHAP values for plotting
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

shap.image_plot(shap_values, to_explain, index_names)

