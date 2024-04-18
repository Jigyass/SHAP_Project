import json
import numpy as np
import torch
from torchvision import models
from torchvision.models import VGG16_Weights
import shap
import pandas as pd
from torch.nn.functional import softmax

# Normalization function
def normalize(image, mean, std):
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

# Load the pretrained model with new weights parameter
model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()

# Download and load the ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)

# SHAP dataset
X, _ = shap.datasets.imagenet50()
X /= 255  # Normalize dataset values

# Constants for normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# SHAP Explainer
explainer = shap.GradientExplainer((model, model.features[7]), normalize(X, mean, std))

# Initialize a dictionary to count feature occurrences
feature_counts = {}
feature_values = {}

# Process each image
for i in range(len(X)):
    print(f'Processing image {i+1}/{len(X)}...')
    to_explain = X[[i]]
    normalized_image = normalize(to_explain, mean, std)
    shap_values, indexes = explainer.shap_values(normalized_image, nsamples=50)
    
    # Flatten SHAP values and iterate through them
    flat_shap_values = shap_values[0][0].flatten()
    for idx, val in enumerate(flat_shap_values):
        if idx not in feature_counts:
            feature_counts[idx] = 0
            feature_values[idx] = []
        if abs(val) > 0.01:  # Threshold for considering a SHAP value significant
            feature_counts[idx] += 1
            feature_values[idx].append(val)

# Create a DataFrame to analyze the results
features_df = pd.DataFrame({
    "Feature Index": feature_counts.keys(),
    "Count": feature_counts.values(),
    "Average SHAP Value": [np.mean(feature_values[idx]) for idx in feature_counts.keys()],
    "Sum SHAP Value": [np.sum(feature_values[idx]) for idx in feature_counts.keys()]
})

# Sort by the most frequently influential features
sorted_features_df = features_df.sort_values(by="Count", ascending=False)

# Save to CSV
sorted_features_df.to_csv("frequent_feature_counts.csv", index=False)

print("Most frequently influential features:", sorted_features_df.head(10))

