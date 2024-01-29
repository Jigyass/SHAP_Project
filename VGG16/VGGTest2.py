from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions
from tensorflow.keras.models import Model
import shap
import numpy as np

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

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# Select an earlier layer for SHAP analysis, for example, 'block3_conv3'
layer_name = 'block3_conv3'
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

# (Optional) Here, you can train or fine-tune your model if needed
# ...

# Prepare a background dataset for SHAP
background_data, _ = test_generator.next()

# Initialize the SHAP explainer with the model instance
explainer = shap.DeepExplainer(model, background_data)

# Compute SHAP values for a subset of your test data
test_images, _ = test_generator.next()
shap_values = explainer.shap_values(test_images)

# Visualize the SHAP values for the first prediction
shap.image_plot(shap_values, -test_images, show=False)

