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
saveModel ="D:\\FinalProject\\\model" 
class_names =[]

def loadmodel():
    global mymodel
    mymodel  = tf.keras.models.load_model('D:\\FinalProject\\model\\ModeltrainV1.keras')
   # mymodel.summary()
    return mymodel


def predict_from_folders(data_dir, class_names):
    img_array =[]
    for class_name in class_names:
        class_folder = os.path.join(data_dir, class_name)
        if os.path.isdir(class_folder):
            print(f"Processing images in folder: {class_folder}")
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                if image_path.endswith(('png', 'jpg', 'jpeg')):
                    
                    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width) )
                    img_arr = tf.keras.utils.img_to_array(img)
             
                    img_array_expanded = tf.expand_dims(img_arr, 0)
                    img_array.append(img_array_expanded)

    return img_array



    #print(class_name)

def showImage(combined_array=[], class_names=[]):
    if len(combined_array) > 0:
        plt.figure(figsize=(10, 10))
        num_images = min(len(combined_array), 9)

    batch_size = 9
    total_images = len(combined_array)
    for start  in range(0, total_images, batch_size):
        end = min(start + batch_size, total_images)
        plt.figure(figsize=(10, 10))
        for i in  range(start, end):
            ax = plt.subplot(3, 3, (i - start) + 1)   
            img, desc = combined_array[i] 
            img = img.numpy().astype("uint8")
            # img = imagearray[i].numpy().astype("uint8")
            if img.shape[0] == 1:  
                img = tf.squeeze(img)  

            plt.imshow(img)
            plt.title(desc)
            plt.axis("off")

    plt.show(block=True) 



def predictModel(dir_path,className =[]):

    try:
        mymodel = loadmodel()
        img_array = []
        descpredicted = []
        for class_name in className:
            class_folder = os.path.join(dir_path, class_name)
            print(class_folder)
            if os.path.isdir(class_folder):
            #     print(f"Processing images in folder: {class_folder}")
                for image_name in os.listdir(class_folder):
                    image_path = os.path.join(class_folder, image_name)
                    print(image_path)
                    if image_path.endswith(('png', 'jpg', 'jpeg')):              
                        img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width) )
                        img_arr = tf.keras.utils.img_to_array(img)
                        img_prediction = tf.expand_dims(img_arr, 0)
                     

                        # img_prediction = tf.keras.utils.img_to_array(img)
                        # img_prediction = tf.expand_dims(img_prediction, 0)

                        predictions = mymodel.predict(img_prediction)
                        print("predictions",predictions[0])
                        score = tf.nn.softmax(predictions[0])
                        predicted_class = className[np.argmax(score)]
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


def GetClassName(path):
    global class_names
    class_names = sorted(os.listdir(path))
    return class_names



def main():
    try:
        dirPathtestdata = "D:\FinalProject\dataTest"
        # _class_names = GetClassName("D:\\FinalProject\\kaggle_datasets1\\FruitsClassification\\train")
        # print(_class_names)
        _class_names = ['Apple' , 'Banana']
        resultPrediction =  predictModel(dirPathtestdata,_class_names)
        if( len(resultPrediction) > 0 ):
            showImage(resultPrediction)
        #loadmodel()
        # imageList = predict_from_folders("D:\FinalProject\dataTest",_class_names)
        # if len(imageList) > 0 :
           
        #     showImage(imageList)
  
       # testdataset = tf.keras.utils.image_dataset_from_directory("D:/FinalProject/dataTest",image_size=(img_height, img_width), batch_size=batch_size)
     

    except Exception as ex:
        print(ex)



if __name__ == '__main__':
    main()