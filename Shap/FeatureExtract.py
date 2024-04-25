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

# List to store data frames
df_list = []

# Process each image
for i in range(len(X)):
    print(f'Processing image {i+1}/{len(X)}...')
    to_explain = X[[i]]
    normalized_image = normalize(to_explain, mean, std)
    shap_values, indexes = explainer.shap_values(normalized_image, ranked_outputs=1, nsamples=50)

    # Each SHAP value array corresponds to a class. For ImageNet, the length of shap_values should be 1 due to ranked_outputs=1
    feature_shap_values = shap_values[0][0]  # Get SHAP values for the top class

    # Flatten SHAP values and store each feature's value
    feature_shap_values_flat = feature_shap_values.flatten()
    temp_df = pd.DataFrame({
        "Image Index": i,
        "Feature Index": range(len(feature_shap_values_flat)),
        "SHAP Value": feature_shap_values_flat
    })
    df_list.append(temp_df)

# Concatenate all dataframes
results_df = pd.concat(df_list, ignore_index=True)

# Save results to CSV
results_df.to_csv("detailed_imagenet_shap_values.csv", index=False)

