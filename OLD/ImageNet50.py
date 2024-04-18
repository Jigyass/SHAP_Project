import json
import numpy as np
import torch
from torchvision import models, transforms
import shap
import matplotlib.pyplot as plt
from torch.nn.functional import softmax

# Normalization function remains the same
def normalize(image, mean, std):
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

# Load the model
model = models.vgg16(pretrained=True).eval()

# SHAP dataset
X, y = shap.datasets.imagenet50()
X /= 255

# ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)

# SHAP Explainer
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
e = shap.GradientExplainer((model, model.features[7]), normalize(X, mean, std))

for i in range(len(X)):
    print(f'Processing image {i+1}/{len(X)}...')
    to_explain = X[[9]]
    normalized_image = normalize(to_explain, mean, std)
    shap_values, indexes = e.shap_values(normalized_image, ranked_outputs=2, nsamples=200)

    # Convert indexes to names
    index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

    # Model prediction for the top classes
    preds = model(normalized_image)
    probs = softmax(preds, dim=1)
    top_probs, top_classes = torch.topk(probs, 2)  # Getting the top two predictions
    predicted_percentage = top_probs.detach().numpy()[0] * 100  # Convert to percentage

    # Adjust SHAP values for plotting
    shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

    # Print the top two predictions along with their percentages
    for j in range(2):
        print(f"Prediction {j+1}: {class_names[str(top_classes[0][j].item())][1]}, {predicted_percentage[j]:.2f}%")

    # Plot the SHAP values for the current image
    shap.image_plot(shap_values, to_explain, index_names)

