import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model
from PIL import Image

def load_and_preprocess_image_from_path(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize the image to the input size expected by VGG16
        img_array = image.img_to_array(img)  # Convert the image to a numpy array
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size
        return preprocess_input(img_array_expanded_dims)  # Preprocess the image
    except Exception as e:
        print(f"Error loading or processing image: {e}")
        return None

def create_feature_extraction_model(layer_names):
    # Load the pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False)
    outputs = [base_model.get_layer(name).output for name in layer_names]
    # Create a new model that will output these layers
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# Specify the layer names from which you want to get outputs
layer_names = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']

# Create the feature extraction model
feature_extraction_model = create_feature_extraction_model(layer_names)

# Path to your image
image_path = '/home/darksst/Desktop/SHAP_Project/test.jpg'

# Load and preprocess the image
processed_image = load_and_preprocess_image_from_path(image_path)

if processed_image is not None:
    # Extract features from the specified layers
    features = feature_extraction_model.predict(processed_image)
    print("Features extracted successfully.")
    for i, feature in enumerate(features):
        print(f"Feature shape from layer {layer_names[i]}:", feature.shape)
else:
    print("Failed to load and preprocess image.")

