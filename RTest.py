import numpy as np
import json
import requests
import shap
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load pre-trained model
model = VGG16(weights='imagenet')

# Define a function to preprocess the images
def model_predict(data):
    # Preprocess input data for the model
    data_preprocessed = preprocess_input(data.copy())
    # Predict with the model
    return model(data_preprocessed)

# Load ImageNet class names directly from the S3 bucket
class_index_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
class_names = json.loads(requests.get(class_index_url).text)

# Define a small set of images to explain
# Replace 'path_to_your_images' with the actual path to your images
images = ['/home/darksst/Desktop/SHAP_Project/test.jpg']
X = np.array([image.img_to_array(image.load_img(img, target_size=(224, 224))) for img in images])

# Create a masker that is used to mask out partitions of the input image
masker = shap.maskers.Image("inpaint_telea", X[0].shape)

# Create an explainer with the model and image masker
explainer = shap.Explainer(model_predict, masker)

# Compute SHAP values (use a small number of samples for speed, adjust as needed)
shap_values = explainer(X, max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])

# Visualization
# Adjust the visualization part based on the error encountered previously.
# Here's a basic example to visualize the first image's explanation
shap.image_plot(shap_values[0], -X[0:1])

