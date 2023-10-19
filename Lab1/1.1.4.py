import numpy as np
import matplotlib.pyplot as plt
import cv2

# ใช้ cv2 อ่านไฟล์ภาพ 2D
image_path = '../images/bird.jpg'
image = cv2.imread(image_path)
plt.imshow(image)

# กำหนดตัวแปรมารับค่ามิติของตัวภาพ
height, width, _ = image.shape

# สร้าง 2D grid โดยใช้ numpy.mgrid()
x, y= np.mgrid[0:height, 0:width]

# กำหนด pixel intensities เป็นค่าความสูงของ 3D surface โดยใช้ R channel
z = image[:, :, 0]

# plot ภาพ 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='gray')  # ใช้ cmap เพื่อกำหนดให้เป็นภาพเฉดสีเทา

# กำหนด label ให้แต่ละแกน
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis (Intensity)')

plt.show()