import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models.vgg import VGG16_Weights
import requests
import json
import numpy as np
import shap

# Load the pre-trained VGG16 model
weights = VGG16_Weights.IMAGENET1K_V1  # or VGG16_Weights.DEFAULT for the most up-to-date
model = models.vgg16(weights=weights)
model.eval()  # Set the model to evaluation mode

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    img = Image.open(image_path)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t

image_path = '/home/darksst/Desktop/SHAP_Project/fruits-360-original-size/fruits-360-original-size/Test/apple_6/r0_259.jpg'
batch_t = preprocess_image(image_path)

# Make a prediction
with torch.no_grad():
    output = model(batch_t)

# Fetch ImageNet class index if not available locally
try:
    with open('imagenet_class_index.json') as f:
        class_idx = json.load(f)
except FileNotFoundError:
    # If local file is not found, fetch from the web
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    class_idx = requests.get(url).json()

idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# Decode the prediction
_, indices = torch.sort(output, descending=True)
percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
for idx in indices[0][:3]:
    print(f"{idx2label[idx]} ({percentage[idx].item():.2f}%)")



e = shap.GradientExplainer((model,model.features[7]), preprocess_image(image_path))
shap_values, indexes = e.shap_values(preprocess_image(image_path), ranked_outputs=2, nsamples=200)

shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

shap.image_plot(shap_values, image_path, idx2label)
