import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from scipy import signal

# Load the VGG16 model
model = VGG16()

# Create a model that outputs the activations of the first convolutional layer
layer_name = 'block1_conv1'  # Name of the first convolutional layer
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
intermediate_layer_model.summary()

# Load and preprocess the image
image_path = 'images/netherland.jpeg'
img = load_img(image_path, target_size=(224, 224))

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Original Image")

img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

plt.subplot(2, 2, 2)
plt.imshow(img[0])
plt.title("Preprocessed Image")
# plt.show()

# Get the feature maps from the first convolutional layer
feature_maps = intermediate_layer_model.predict(img)

# Display the feature maps
num_filters = feature_maps.shape[3]
plt.figure(figsize=(16, 16))
for i in range(num_filters):
    plt.subplot(8, 8, i + 1)
    plt.imshow(feature_maps[0, :, :, i], cmap='gray')
    plt.axis('off')

plt.show()