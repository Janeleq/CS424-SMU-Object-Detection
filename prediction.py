from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
loaded_model = tf.keras.models.load_model('smu_image_classifier.h5')

# Load an image for prediction
img_path = 'C:\wamp64\www\yolov8-cpu\data\smu'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale pixel values

# Make predictions
predictions = loaded_model.predict(img_array)

# Check the prediction
if predictions[0][0] > 0.5:
    print("The image is from SMU.")
else:
    print("The image is not from SMU.")
