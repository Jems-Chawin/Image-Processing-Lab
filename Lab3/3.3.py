import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.models import Model
import keras as keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from numpy import expand_dims
from scipy.signal import convolve2d
img_path = './images/bird.jpg'
img = image.load_img(img_path)
img = image.img_to_array(img)
print(img.shape)
img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
print(img.shape)
img = cv2.resize(img[0],(244,244))
print(img.shape)
imgB,imgG,imgR = cv2.split(img)
img_mean = [123.68, 116.779, 103.939]
meanB = img_mean[2]
meanG = img_mean[1]
meanR = img_mean[0]
img_mean = [meanB,meanG,meanR]
imgB -= meanB
imgG -= meanG
imgR -= meanR
img = cv2.merge([imgB,imgG,imgR])

img_result = np.zeros(img.shape)
model = VGG16(weights = 'imagenet', include_top = False)
model.summary()
kernels, biases = model.layers[1].get_weights()
feature_maps = np.zeros((img.shape[0], img.shape[1], kernels.shape[3]))
for i in range(kernels.shape[3]):
    img_result1 = convolve2d(img[:,:,0],kernels[:,:,0,i],mode="same",boundary="fill",fillvalue=0)     
    img_result2 = convolve2d(img[:,:,1],kernels[:,:,1,i],mode="same",boundary="fill",fillvalue=0)  
    img_result3 = convolve2d(img[:,:,2],kernels[:,:,2,i],mode="same",boundary="fill",fillvalue=0)  
    feature_maps[:, :, i] = img_result1 + img_result2 + img_result3

feature_maps = np.maximum(0,feature_maps)
for i in range(feature_maps.shape[2]):
    plt.subplot(8,8,i+1)
    plt.imshow(feature_maps[:,:,i])
    plt.axis("off")
plt.show()