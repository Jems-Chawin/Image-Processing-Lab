import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image and convert to RGB
img_bgr = cv2.imread("images/galaxy.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(img_rgb)

# Equalize Histogram
eq_r = cv2.equalizeHist(r)
eq_g = cv2.equalizeHist(g)
eq_b = cv2.equalizeHist(b)

eq_img = cv2.merge((eq_r, eq_g, eq_b))

# Create a single figure with multiple subplots
plt.figure(figsize=(14, 12))

# Original image and histogram subplot
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")

plt.subplot(2, 2, 2)
for i, color in enumerate(('r', 'g', 'b')):
    hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
plt.title("Hist Original")

# Equalized image and histogram subplot
plt.subplot(2, 2, 3)
plt.imshow(eq_img)
plt.title("Equalized Image")

plt.subplot(2, 2, 4)
for i, color in enumerate(('r', 'g', 'b')):
    hist = cv2.calcHist([eq_img], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
plt.title("Hist Equalized")

plt.tight_layout()  # Adjust spacing between subplots
plt.show()
