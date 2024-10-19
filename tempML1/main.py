from trainModelTF import TrainDataSet
import os

saveModelPath = "D:\FinalProject\model"
img_height = 180
img_width = 180
batch_size = 32


def GetClassName(path):
    global class_names
    class_names = sorted(os.listdir(path))
    return class_names


def convert(lst):
    res_dict = {}
    for i in range(0, len(lst), 1):
        res_dict[i + 1] = lst[i]

    return res_dict


def testModel():
    modelPath = "D:\FinalProject\model\FruitsClassificationModeltrainV1.keras"
    testPath = "D:\FinalProject\dataTest"
    tr = TrainDataSet()
    classNames = TrainDataSet.load_class_names(
        "D:\\FinalProject\\dataTest"
    )
   
    # classNamesTuple =  convert(classNames)
    print(classNames)
    # classNames = ['Apple','Banana']
    result = tr.predictModel(modelPath,testPath,classNames)
    tr.ShowImage(result)

def groupClassName(classNames):
    newclassName = []
    for item in classNames:
        found = False
        classname = item[0:4].lower().strip()

        for group in newclassName:
            groupName = group[0:4].lower().strip()

            if groupName.startswith(classname):
                found = True
                break

        if not found:
            newclassName.append(item.split()[0])

    print(newclassName)
    return newclassName


def traintmodelNewByShreya():
    dirPath = "D:\\FinalProject\\dataset\\shreyafruits\\train"
    tr = TrainDataSet()
    classNames = TrainDataSet.load_class_names(dirPath)
 
    train_ds, val_ds = tr.TraingDatasetByKeras2(
        dirPath, img_height, img_width, batch_size
    )
    print(train_ds.class_names)
    numClass = len(train_ds.class_names)
    model =  tr.createModel(numClass)
    epochs=10
    print(model.summary())

    model.fit(train_ds,  validation_data=val_ds, epochs=epochs)
    pathModel = saveModelPath+"\ModelShreyafruits.keras"
    model.save(pathModel)
def tainModelFruitsClassification():
    
    tr = TrainDataSet()
    dirPath ="D:\\FinalProject\\kaggle_datasets1\\FruitsClassification2\\train"
    train_ds ,val_ds =  TrainDataSet.TraingDatasetByKeras2(dirPath,img_height, img_width, batch_size)
    class_names = train_ds.class_names
    numClass = len(class_names)
    model =  tr.createModel(numClass)
    model.fit(train_ds,  validation_data=val_ds, epochs=10)
    pathModel = saveModelPath+"\FruitsClassificationModeltrainV1.keras"
    model.save(pathModel)


def main():
    try:

       
        testModel()

        # dirPath = "D:\\FinalProject\\Fruits_360Datasetzip\\fruits-360_dataset\\fruits-360\\Training"
      
        # print(class_names)
     
        # epochs=10
        # print(model.summary())

    

    except Exception as e:
        print(f"Error : {e}")


if __name__ == "__main__":
    main()
