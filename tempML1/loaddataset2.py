import os
from os import listdir
from PIL import Image as PImage
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


batch_size = 32
img_height = 180
img_width = 180
saveModelPath ="D:\FinalProject\model" 
class_names =[]



def loadDataset(path):
    global class_names
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return [], [], []
    
    print(f"Path exists: {path}")
    class_names = sorted(os.listdir(path))
    print(f"  class_names: {class_names}")
    
    images = []
    labels = []
    
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(path, class_name)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            try:
                with PImage.open(image_path) as image:
                    image = image.resize((180, 180))  # Resize image
                    image = image.convert('RGB')  # Convert to RGB
                    images.append(np.array(image))
                    labels.append(label)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    
    return images, labels, class_names


def createDatasetByKeras(path,typedata = "training"):
    try:
      dataset =   tf.keras.utils.image_dataset_from_directory(
                    path,
                    validation_split=0.2,
                    subset=typedata,
                    seed=123,
                    image_size=(img_height, img_width),
                    batch_size=batch_size)
  
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None, None
    
    return dataset
 


def createDataset(path):
    try:
        images, labels, class_names = loadDataset(path)
        images_array = np.array(images, dtype=np.float32) / 255.0  # Normalize images
        labels_array = np.array(labels)
        dataset = tf.data.Dataset.from_tensor_slices((images_array, labels_array))
        dataset = dataset.batch(batch_size)
        return dataset, class_names
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None, None

def createModel(num_classes):
    model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes)
            ])
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    return model




def main():
    try:

        # image = PImage.open("D:/FinalProject/testAI.jpg")
        # image.show()
       
        # checkpoint_callback = ModelCheckpoint(
        #     saveModelPath,
        #     save_best_only=True,  
        #     monitor='val_loss', 
        #     mode='min'
        # )
        
        train_path = "D:\\FinalProject\\kaggle_datasets1\\FruitsClassification\\train"
        valid_path = "D:\\FinalProject\\kaggle_datasets1\\FruitsClassification\\valid"
        
        # traindata, train_class_names = createDataset(train_path)
        # validatedata, valid_class_names = createDataset(valid_path)
        
        # if traindata is None or validatedata is None:
        #     print("Failed to load datasets.")
        #     return
        
        # for image_batch, label_batch in validatedata.take(1):  # Check data shapes
        #     print(f"Image batch shape: {image_batch.shape}")
        #     print(f"Label batch shape: {label_batch.shape}")
        

        # classNum = len(class_names)
        # model = createModel(num_classes=classNum)
        # model.fit(traindata, validation_data=validatedata, epochs=2)
    
        train_ds = createDatasetByKeras(train_path,"training")
        val_ds = createDatasetByKeras(valid_path,"validation")
        print(train_ds.class_names)

        # for image_batch, labels_batch in train_ds:
        #     print(image_batch.shape)
        #     print(labels_batch.shape)
        #     break

        model = createModel(num_classes=len(train_ds.class_names))
        model.fit(train_ds,  validation_data=val_ds, epochs=10)
        pathModel = saveModelPath+"\ModeltrainV1.keras"
        model.save(pathModel)

        # test_path = '/content/drive/MyDrive/FruitsClassification/test'
        # test_ds = createDatasetByKeras(test_path)
    
        
    except Exception as e:
        print(f"Error in main: {e}")



if __name__ == '__main__':
    main()



