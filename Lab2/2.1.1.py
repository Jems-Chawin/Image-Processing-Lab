import random
import cv2
import numpy as np
import math

def get_linear(a, b, img):
    l = []
    for i in range(20):
        g = a[i] * img.copy() + b[i]
        l.append(np.clip(g, 0, 255).astype(np.uint8))
    return l

if __name__ == '__main__':
    img = cv2.imread('images/bird.jpg')

    a = []
    b = []

    for i in range(-5, 15):
        a.append(i + 0.5)
        b.append(i + 1.75)

    linear_img = get_linear(a, b, img)

    # Get image dimensions
    height, width, _ = linear_img[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('2_1_1.mp4', fourcc, 5.0, (width, height))

    # Write each frame to the video
    for i in range(20):
        out.write(linear_img[i])

    # Release the VideoWriter
    out.release()

    print("Video saved as linear_images.mp4")
