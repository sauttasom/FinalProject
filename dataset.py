
import os
import numpy as np
from glob import glob
from tqdm import tqdm

# Define class names and number of images per class
class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']

root_path = 'D:\\FinalProject\\kaggle_datasets1\\FruitsClassification'
# Define train, valid, and test directories and create them if they don't exist
train_dir = "./train"
valid_dir = "./valid"
test_dir  = "./test"



def loaddataset():
    global n_images_per_class 
    for class_name in class_names:
        folder_path = os.path.join(root_path, class_name)
        if os.path.exists(folder_path):
            print(f"Folder '{folder_path}' exists.")
        
            n_images_per_class = len(os.listdir(folder_path))
            print(f"Number of images in '{folder_path}': {n_images_per_class}")
        else:
            print(f"Folder '{folder_path}' does not exist.")




for directory in [train_dir, valid_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create subdirectories for each class in train, valid, and test directories
for name in class_names:
    for directory in [train_dir, valid_dir, test_dir]:
        class_path = os.path.join(directory, name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)


# Collect all image paths for each class
all_class_paths = [glob(f"./{name}/*") for name in class_names]

# Define training, validation, and testing size
total_size = sum([len(paths) for paths in all_class_paths])

train_ratio = 0.97
valid_ratio = 0.02
test_ratio  = 0.01

train_size = int(total_size * train_ratio)
valid_size = int(total_size * valid_ratio)
test_size  = int(total_size * test_ratio)

train_images_per_class = int(n_images_per_class * train_ratio)
valid_images_per_class = int(n_images_per_class * valid_ratio)
test_images_per_class  = int(n_images_per_class * test_ratio)

print("Total Data Size  :   {}".format(total_size))
print("Training Size    :   {}".format(train_size))
print("Validation Size  :   {}".format(valid_size))
print("Testing Size     :   {}\n".format(test_size))

