import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import tensorflow as tf

# Load the VGG16 model
model = VGG16()

# Load and preprocess the image
image_path = 'C:\\Users\\Jems\\Desktop\\netherland.jpeg'
img = load_img(image_path, target_size=(224, 224))

# Reshape the image to 4D array
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Reshape to (1, H, W, Ch)

# Resize the image to match VGG16 input size
img_resized = cv2.resize(img_array[0], (224, 224))  # Resize the first image in the batch

# Subtract dataset mean (mean subtraction)
img_mean = [123.68, 116.779, 103.939]  # Mean values for ImageNet dataset
img_mean_shifted = img_resized - img_mean

# Display the preprocessed image
plt.imshow(img_mean_shifted.astype(np.uint8))  # Convert to uint8 before displaying
plt.axis('off')
plt.show()
