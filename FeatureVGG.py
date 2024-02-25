import os
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Load the VGG16 model
model = VGG16()
# Restructure the model to remove the final layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# Print model summary
print(model.summary())

features = {}
directory = '/home/darksst/Desktop/SHAP_Project/'

# Ensure the directory path ends with a slash
if not directory.endswith('/'):
    directory += '/'

for image in os.listdir(directory):
    # Check if the file is an image (for simplicity, check for jpg extension)
    if image.lower().endswith('.jpg'):
        image_path = os.path.join(directory, image)
        
        # Load the image with target size of 224x224
        img = load_img(image_path, target_size=(224, 224))
        # Convert the image to a numpy array
        img = img_to_array(img)
        # Reshape the image to fit the model input
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        # Preprocess the image
        img = preprocess_input(img)
        # Extract features
        feature = model.predict(img, verbose=0)
        # Store the feature
        features[image] = feature

# Example to access a feature for 'test.jpg'
print(features.get('test.jpg'))

# Save the features dictionary to a file using pickle
with open('/home/darksst/Desktop/SHAP_Project/features.pkl', 'wb') as file:
    pickle.dump(features, file)

