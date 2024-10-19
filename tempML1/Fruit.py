import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam



print(tf.__version__)


root_path = "D:\\FinalProject\\dataset"
class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']


train_path = os.path.join(root_path, 'train')
test_path = os.path.join(root_path, 'test')


os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
train_ratio = 0.8

for class_name in class_names:
    # สร้างโฟลเดอร์สำหรับแต่ละคลาสใน train และ test
    os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_path, class_name), exist_ok=True)
  
    class_folder =   os.path.join("D:", os.sep, "FinalProject", "dataset",class_name)
    all_files =os.listdir(class_folder)
    #os.listdir('D:\\FinalProject\\dataset\\Apple')

    random.shuffle(all_files)
    train_count = int(len(all_files) * train_ratio)

    train_files = all_files[:train_count]
    test_files = all_files[train_count:]

    for file in train_files:
        src_file = os.path.join(class_folder, file)
        dst_file = os.path.join(train_path, class_name, file)
        shutil.move(src_file, dst_file)

    for file in test_files:
        src_file = os.path.join(class_folder, file)
        dst_file = os.path.join(test_path, class_name, file)
        shutil.move(src_file, dst_file)


batch_size = 32
img_height = 150
img_width = 150
epochs = 10
learning_rate = 0.001

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# ฝึกโมเดล
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples ,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples 
)

# # ประเมินโมเดลt
# loss, accuracy = model.evaluate(test_generator)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")