from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('data/image_classification_smu/training/', target_size=(200,200), batch_size = 3, class_mode = 'binary')
validation_dataset = train.flow_from_directory('data/image_classification_smu/validation/', target_size=(200,200), batch_size = 3, class_mode = 'binary')

print(len(train_dataset.classes))
print(train_dataset.class_indices)

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    # tf.keras.layers.Dropout(0.2), #possibly see how it fare
                                    tf.keras.layers.Dense(1, activation = 'sigmoid')])

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch = 10,
                      epochs = 5,
                      validation_data = validation_dataset)


# model.save('smu_image_classifier')

dir_path = 'data/image_classification_smu/test'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path + '//' + i, target_size = (200,200))
    plt.imshow(img)
    plt.show()

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis = 0)
    images = np.vstack([X])
    val = model.predict(images)
    print(val)
    if val == 0:
        print("This image is from smu")
    else:
        print("This image is not from smu")