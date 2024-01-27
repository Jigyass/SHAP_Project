from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# Dataset paths
train_dir = '/home/darksst/Desktop/SHAP_Project/fruits-360-original-size/fruits-360-original-size/Training'
validation_dir = '/home/darksst/Desktop/SHAP_Project/fruits-360-original-size/fruits-360-original-size/Validation'
test_dir = '/home/darksst/Desktop/SHAP_Project/fruits-360-original-size/fruits-360-original-size/Test'

# ImageDataGenerators for each set
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load and iterate datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Use the test generator to make predictions
# We will take a batch of images from your test set
for batch, _ in test_generator:
    # Make predictions
    predictions = model.predict(batch)

    # Decode predictions
    decoded_predictions = decode_predictions(predictions, top=3)

    # Print the predictions for each image in the batch
    for i, preds in enumerate(decoded_predictions):
        print(f"Image {i + 1}:")
        for imagenet_id, label, score in preds:
            print(f"   {label} ({score:.2f})")

    # Break after first batch to avoid predicting on the entire dataset
    break

import shap
import numpy as np

# Load a subset of your data for SHAP analysis
# For example, let's use a batch from your test generator
for images, _ in test_generator:
    data_for_shap = images
    break

# Initialize the SHAP explainer
# model is your trained VGG model
explainer = shap.DeepExplainer(model, data_for_shap)

# Compute SHAP values
shap_values = explainer.shap_values(data_for_shap)

# Visualize the SHAP values for the first prediction
# Adjust the index to see other predictions
shap.image_plot(shap_values, -data_for_shap, show=False)

