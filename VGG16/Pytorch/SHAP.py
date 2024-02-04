import torch
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import VGG16_Weights
import numpy as np
import shap
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root='/home/darksst/Desktop/SHAP_Project/fruits-360-original-size/fruits-360-original-size/Training', transform=transform)
val_dataset = datasets.ImageFolder(root='/home/darksst/Desktop/SHAP_Project/fruits-360-original-size/fruits-360-original-size/Validation', transform=transform)
test_dataset = datasets.ImageFolder(root='/home/darksst/Desktop/SHAP_Project/fruits-360-original-size/fruits-360-original-size/Test', transform=transform)

# Data loaders with limited images for SHAP analysis
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)  # Limited to 2 images

# Load the pre-trained VGG16 model
model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)

# Example of making predictions with the model
model.eval()
with torch.no_grad():
    for batch, _ in test_loader:
        batch = batch.to(device)
        outputs = model(batch)
        # Get the top 3 predictions
        _, top_preds = torch.topk(outputs, 3, dim=1)
        for i, preds in enumerate(top_preds):
            print(f"Image {i+1}:")
            for idx in preds:
                print(f"   Class Index: {idx.item()}")
        break  # Process only the first batch of 2 images

# SHAP analysis
model.eval()
data_for_shap, _ = next(iter(test_loader))
data_for_shap = data_for_shap.to(device)

# Initialize the SHAP explainer
explainer = shap.GradientExplainer((model, model.features[7]), data_for_shap)

# Compute SHAP values
shap_values, indexes = explainer.shap_values(data_for_shap, ranked_outputs=2, nsamples=50)

# Visualize the SHAP values
shap.image_plot(shap_values, -data_for_shap.cpu().numpy())

