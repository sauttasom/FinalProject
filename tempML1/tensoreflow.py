
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt


 
def loaddataset():
    global data_dir

    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
  
    archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
    data_dir = pathlib.Path(archive).with_suffix('')

    # roses = list(data_dir.glob('roses/*'))
    # image = PIL.Image.open(str(roses[0]))

    # plt.imshow(image)
    # plt.axis('off')  # Hide axis labels
    # plt.show()
def setLabel():
    global train_ds
    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    

def main():
   try:
    loaddataset()
    setLabel()
    print('train_ds')
    print(train_ds)
    class_names = train_ds.class_names
    print(class_names)

    for images, labels in train_ds.take(1):  
        for i in range(len(images)):
            print(f"Image {i+1}:")
            print(f"  Label: {labels[i]} ({class_names[labels[i]]})")



   except ValueError:
    print("Conversion failed. Input is not a valid float.") 

if __name__ == '__main__':
    main()