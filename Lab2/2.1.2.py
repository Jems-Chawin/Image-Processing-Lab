import cv2
import numpy as np
from matplotlib import pyplot as plt
image1 = cv2.imread("images/bird.jpg")
image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('2_1_2.mp4', fourcc, 2, (1280, 720))
image1 = image1.astype(float)

image1 = cv2.resize(image1,(1280,720))
a_factor = 1
b_factor = 0
gramma_factor = 0

steps_gramma_factor = 0.1

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 0)  # blue color
thickness = 2
position = (100, 100) 
num_levels = 255

for i in range(20) :
    result = a_factor * pow(image1,gramma_factor) + b_factor
    if(gramma_factor == 0) :
        quantized_image = result.astype(np.uint8)
    else :
        blue_channel, green_channel, red_channel = cv2.split(result)

        min_blue = np.min(blue_channel)
        max_blue = np.max(blue_channel)
        min_green = np.min(green_channel)
        max_green = np.max(green_channel)
        min_red = np.min(red_channel)
        max_red = np.max(red_channel)

        step_size_blue = (max_blue - min_blue) / num_levels
        step_size_green = (max_green - min_green) / num_levels
        step_size_red = (max_red - min_red) / num_levels

        quantized_blue = ((blue_channel - min_blue) / step_size_blue).astype(np.uint8)
        quantized_green = ((green_channel - min_green) / step_size_green).astype(np.uint8)
        quantized_red = ((red_channel - min_red) / step_size_red).astype(np.uint8)
        quantized_image = cv2.merge((quantized_blue, quantized_green, quantized_red))

    cv2.putText(quantized_image, "Gramma factor : " + "{:.1f}".format(gramma_factor), position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    video_writer.write(quantized_image)
    
    gramma_factor += steps_gramma_factor

video_writer.release()