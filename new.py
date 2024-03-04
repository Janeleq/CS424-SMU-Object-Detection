from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


img = image.load_img("data/image_classification_smu/training/smu/image1.jpg")
print(plt.imshow(img))
plt.show()
# height x width x RGB 3 colors
print(cv2.imread("data/image_classification_smu/training/smu/image1.jpg").shape)

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('data/image_classification_smu/training/', target_size=(200,200), batch_size = 3, class_mode = 'binary')
validation_dataset = train.flow_from_directory('data/image_classification_smu/validation/', target_size=(200,200), batch_size = 3, class_mode = 'binary')
print(train_dataset)
print(train_dataset.classes)
print(train_dataset.class_indices)

model = tf.keras.models.sequential([tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.flatten(),
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    tf.keras.layers.Dense(1, activation = 'sigmoid')])

model.compile(loss = 'binary_crossentropy',
              optimizer = RMSprop(lr=0.001),
              metrics = ['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch = 3,
                      epochs = 10,
                      validation_data = validation_dataset)