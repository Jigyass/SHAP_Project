import os
import shutil

# Define the path to your test dataset directory
test_dir = '/home/darksst/Desktop/SHAP_Project/Data/tiny-imagenet-200/test'

# Iterate over all class directories in the test directory
for class_dir in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_dir)
    if os.path.isdir(class_path):
        images_path = os.path.join(class_path, 'images')
        if os.path.exists(images_path):
            for img_name in os.listdir(images_path):
                shutil.move(os.path.join(images_path, img_name), os.path.join(class_path, img_name))
            os.rmdir(images_path)  # Remove the now-empty 'images' subdirectory

