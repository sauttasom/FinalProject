import os
from os import listdir
from PIL import Image as PImage
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


class TrainDataSet:
    batch_size = 32
    img_height = 180
    img_width = 180
    saveModelPath = "D:\FinalProject\model"
    train_ds = tuple[None, None]
    val_ds = tuple[None, None]
    _epochs = 10
    # def __init__(self, path):
    #     self.__loadDataset(path)

    def loadmodel(self,path):
        model = tf.keras.models.load_model(path)
        return model

    @staticmethod
    def load_class_names(path):
        return sorted(entry.name for entry in os.scandir(path) if entry.is_dir())
    
    def __predicttionOneImage(self, model, imagePath, classNames=[]):
        img = tf.keras.utils.load_img(  imagePath, target_size=(self.img_height, self.img_width))
        img_arr = tf.keras.utils.img_to_array(img)
        img_prediction = tf.expand_dims(img_arr, 0)
        predictions = model.predict(img_prediction)
        print("predictions", predictions[0])
        score = tf.nn.softmax(predictions[0])
                          

        maxscore_index = np.argmax(score)  
        maxscore_value = predictions[0][maxscore_index]
                            
        print("Max Score Index: ", maxscore_index)
        print("Max Score Value: ", maxscore_value)
        
    def __predicttionDataTEST(self, model, dir_path, classNames=[]):
        try:
            combined_array  = list[tuple]
            img_array = []
            descpredicted = []
            rootPath = ''
            if os.path.isdir(dir_path):
                rootPath = dir_path
                for folder in os.listdir(dir_path):
                    print(os.path.join(rootPath, folder))
                    class_folder  =os.path.join(rootPath, folder)
                    for image_name in os.listdir(class_folder):
                        print(image_name)
                        if image_name.endswith(("png", "jpg", "jpeg")):
                            imagePath = os.path.join(class_folder, image_name)
                            print(imagePath)
                            img = tf.keras.utils.load_img(  imagePath, target_size=(self.img_height, self.img_width))
                            img_arr = tf.keras.utils.img_to_array(img)
                            img_prediction = tf.expand_dims(img_arr, 0)

                                # img_prediction = tf.keras.utils.img_to_array(img)
                                # img_prediction = tf.expand_dims(img_prediction, 0)

                            predictions = model.predict(img_prediction)
                            print("predictions", predictions[0])
                            score = tf.nn.softmax(predictions[0])
                          

                            maxscore_index = np.argmax(score)  
                            maxscore_value = predictions[0][maxscore_index]
                            
                            print("Max Score Index: ", maxscore_index)
                            print("Max Score Value: ", maxscore_value)
                            # predicted_class = classNames[np.argmax(score)]
                            # result = (  
                            #     f"Image Name = {os.path.basename(image_name).split('/')[-1]}\n"
                            #     f"Predicted Result = {predicted_class}"
                            # )

                            # img_array.append(img_prediction)
                            # descpredicted.append(result)
                            combined_array = list(zip(img_array, descpredicted))
                           
                        else:
                            print(f"{dir_path} is not a directory.")

            
            return combined_array
        except Exception as ex:
            print(ex)


    def __predicttionDataTESTMulitFolder(self, model, dir_path, classNames=[]):

        try:
            combined_array  = list[tuple]
            img_array = []
            descpredicted = []
            for class_name in classNames:
                class_folder = os.path.join(dir_path, class_name)
                print(f"folder : {class_folder}")
                if os.path.isdir(dir_path):
                    print(f"Processing images in folder: {class_folder}")
                    for image_name in os.listdir(class_folder):
                        image_path = os.path.join(class_folder, image_name)
                        print(image_path)
                        if image_path.endswith(("png", "jpg", "jpeg")):
                            img = tf.keras.utils.load_img(
                                image_path, target_size=(self.img_height, self.img_width)
                            )
                            img_arr = tf.keras.utils.img_to_array(img)
                            img_prediction = tf.expand_dims(img_arr, 0)

                            # img_prediction = tf.keras.utils.img_to_array(img)
                            # img_prediction = tf.expand_dims(img_prediction, 0)

                            predictions = model.predict(img_prediction)
                            print("predictions", predictions[0])
                            score = tf.nn.softmax(predictions[0])
                            predicted_class = classNames[np.argmax(score)]
                            result = (
                                f"Class Name = {class_name}\n"
                                f"Image Name = {os.path.basename(image_path).split('/')[-1]}\n"
                                f"Predicted Result = {predicted_class}"
                            )

                            img_array.append(img_prediction)
                            descpredicted.append(result)
                            combined_array = list(zip(img_array, descpredicted))

            return combined_array

        except Exception as ex:
            print(ex)
   
    def predictModel(self, modelpath, dirtest_path, className=[]):
        try:
            if modelpath != "":
                model = tf.keras.models.load_model(modelpath)
                if model != None:
                    
                    print(model.summary())
                 #   resultPrediction = self.__predicttionOneImage(model, dirtest_path, className)
                    resultPrediction = self.__predicttionDataTESTMulitFolder(model, dirtest_path, className)
                    return resultPrediction
                else:
                    print("Not Found Model")

        except Exception as ex:
            print(ex)
   
    @staticmethod
    def TraingDatasetByKeras2(data_dir, img_height, img_width, batch_size):
        val_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(img_height,img_width),
                batch_size=batch_size)
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(img_height, img_width),
                batch_size=batch_size)
        
        return train_ds,val_ds
    
    def ShowImage(self ,perdictionResutl = list[tuple]):

        if len(perdictionResutl) > 0:
            batch_size = 9
            total_images = len(perdictionResutl)
            for start  in range(0, total_images, batch_size):
                end = min(start + batch_size, total_images)
                plt.figure(figsize=(10, 10))
                for i in  range(start, end):
                    ax = plt.subplot(3, 3, (i - start) + 1)   
                    img, desc = perdictionResutl[i] 
                    img = img.numpy().astype("uint8")
                    # img = imagearray[i].numpy().astype("uint8")
                    if img.shape[0] == 1:  
                        img = tf.squeeze(img)  

                    plt.imshow(img)
                    plt.title(desc)
                    plt.axis("off")

            plt.show(block=True) 


    
    def TraingDatasetByKeras(self, pathTrain, pathValidate, pathSaveModel):
        try:
            train_ds = self.__createDatasetByKeras(pathTrain, "training")
            val_ds = self.__createDatasetByKeras(pathValidate, "validation")

            return train_ds,val_ds
            # if train_ds.count > 0 and val_ds.count > 0:
            #     numClass = len(train_ds.class_names)
            #     if numClass > 0:
            #         model = self.__createModel(num_classes=numClass)
            #         model.fit(train_ds, validation_data=val_ds, epochs=self._epochs)
            #         model.save(pathSaveModel)
            #         return model
            #     else:
            #         return
            # else:
            #     return
        except Exception as e:
            print(f"Error : {e}")

    def __createDatasetByKeras(self, path, typedata="training"):
        try:
            dataset = tf.keras.utils.image_dataset_from_directory(
                path,
                validation_split=0.2,
                subset=typedata,
                seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
            )
            return dataset
        except Exception as e:
            print(f"Error creating dataset: {e}")
            return None, None

    def createModel(self ,num_classes):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(num_classes),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        return model


