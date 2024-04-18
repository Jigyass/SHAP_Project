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

# Initialize an array to accumulate SHAP values for averaging
aggregate_shap_values = None

# Process each image
for i in range(len(X)):
    print(f'Processing image {i+1}/{len(X)}...')
    to_explain = X[[i]]
    normalized_image = normalize(to_explain, mean, std)
    shap_values, indexes = explainer.shap_values(normalized_image, ranked_outputs=1, nsamples=50)

    # Aggregate SHAP values
    if aggregate_shap_values is None:
        aggregate_shap_values = np.abs(shap_values[0]).reshape(1, -1)  # Initialize the aggregate values
    else:
        aggregate_shap_values += np.abs(shap_values[0]).reshape(1, -1)  # Sum SHAP values

# Compute the average of SHAP values across all images
average_shap_values = aggregate_shap_values / len(X)

# Create a DataFrame for analysis
features_df = pd.DataFrame({
    "Feature Index": range(average_shap_values.shape[1]),
    "Average SHAP Value": average_shap_values.flatten()
})

# Sort by the most influential features
sorted_features_df = features_df.sort_values(by="Average SHAP Value", ascending=False)

# Save to CSV
sorted_features_df.to_csv("average_shap_values.csv", index=False)

# Optionally, you can also visualize the top positive and negative features
top_features = sorted_features_df.head(10)
bottom_features = sorted_features_df.tail(10)

print("Top Positive Features:", top_features)
print("Top Negative Features:", bottom_features)

