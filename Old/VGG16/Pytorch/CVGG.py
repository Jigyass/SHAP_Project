from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models.vgg import VGG16_Weights
import shap
import requests
import json

# Load the pre-trained VGG16 model
weights = VGG16_Weights.IMAGENET1K_V1
model = models.vgg16(weights=weights)
model.eval()

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
    return batch_t, np.array(img)

image_path = '/home/darksst/Desktop/SHAP_Project/fruits-360-original-size/fruits-360-original-size/Test/apple_6/r0_259.jpg'
batch_t, np_img = preprocess_image(image_path)

# Fetch ImageNet class index if not available locally
try:
    with open('imagenet_class_index.json') as f:
        class_idx = json.load(f)
except FileNotFoundError:
    # If local file is not found, fetch from the web
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    class_idx = requests.get(url).json()

idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# Make a prediction
with torch.no_grad():
    output = model(batch_t)

# Assuming class_idx and idx2label are already defined as in your original code

# Decode the prediction
_, indices = torch.sort(output, descending=True)
percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
for idx in indices[0][:3]:
    print(f"{idx2label[idx]} ({percentage[idx].item():.2f}%)")

# SHAP explanation
e = shap.GradientExplainer((model, model.features[7]), batch_t)
shap_values, indexes = e.shap_values(batch_t, ranked_outputs=2, nsamples=200)

# Convert SHAP values to match the image format
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

# For shap.image_plot, convert your image to a format that can be accepted
# Ensure the numpy image is in the correct dimension for plotting
# If your image is grayscale, you might need to repeat it across three channels
np_img_rgb = np_img if np_img.ndim == 3 else np.repeat(np_img[:, :, np.newaxis], 3, axis=2)

# Adjust the numpy image shape if necessary
np_img_rgb = np_img_rgb.reshape((1,) + np_img_rgb.shape)  # Adding a batch dimension

# Convert indexes to a list of integers if it's a tensor
if torch.is_tensor(indexes):
    indexes = indexes.cpu().numpy()  # First ensure it's moved to CPU and converted to a numpy array

# Now indexes should be an array, you can safely use it to index idx2label
labels = [idx2label[i] for i in indexes.flatten().tolist()]  # Flatten and convert to list if necessary


shap.image_plot(shap_values, -np_img_rgb, [idx2label[i] for i in indexes])

