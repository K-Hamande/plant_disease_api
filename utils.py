import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.image import resize

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
