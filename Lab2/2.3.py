import cv2
import matplotlib.pyplot as plt
import numpy as np

image = plt.imread('images/galaxy.jpg')
template = plt.imread('images/beach.jpg')


# ------------------------------------------ r,g,b image ----------------------------------------- #
b_image, g_image, r_image = cv2.split(image)

b_template, g_template, r_template = cv2.split(template)


# ------------------------------------------ norm image ------------------------------------------ #
b_norm_image = (cv2.calcHist([b_image], [0], None, [256], [0, 256]))
b_norm_image /= b_norm_image.sum()    
cdf_b_image = b_norm_image.cumsum()

g_norm_image = (cv2.calcHist([g_image], [0], None, [256], [0, 256]))
g_norm_image /= g_norm_image.sum()    
cdf_g_image = g_norm_image.cumsum()

r_norm_image = (cv2.calcHist([r_image], [0], None, [256], [0, 256]))
r_norm_image /= r_norm_image.sum()    
cdf_r_image = r_norm_image.cumsum()


# ----------------------------------------- norm template ---------------------------------------- #
b_norm_template = (cv2.calcHist([b_template], [0], None, [256], [0, 256]))
b_norm_template /= b_norm_template.sum()    
cdf_b_template = b_norm_template.cumsum()
print(cdf_b_template)

g_norm_template = (cv2.calcHist([g_template], [0], None, [256], [0, 256]))
g_norm_template /= g_norm_template.sum()    
cdf_g_template = g_norm_template.cumsum()

r_norm_template = (cv2.calcHist([r_template], [0], None, [256], [0, 256]))
r_norm_template /= r_norm_template.sum()    
cdf_r_template = r_norm_template.cumsum()


# ----------------------------------------- matching cdf ----------------------------------------- #
def histogram_matching_index(cdf_template, cdf_original):
    lut = []
    for cdf_ori in cdf_original:
        diff = np.abs(cdf_ori - cdf_template)
        min_diff_idx = np.argmin(diff)
        lut.append(min_diff_idx)
    return lut

lut_r = histogram_matching_index(cdf_r_template, cdf_r_image)
lut_g = histogram_matching_index(cdf_g_template, cdf_g_image)
lut_b = histogram_matching_index(cdf_b_template, cdf_b_image)

# --------------------------------------- เข้า lookup table เพื่อ matching -------------------------------------- #
matched_r = cv2.LUT(r_image, np.array(lut_r, dtype=np.uint8))
matched_g = cv2.LUT(g_image, np.array(lut_g, dtype=np.uint8))
matched_b = cv2.LUT(b_image, np.array(lut_b, dtype=np.uint8))

matched_image = cv2.merge((matched_b,matched_g,matched_r))


# ----------------------------------------- norm matching ---------------------------------------- #
b_norm_matching = (cv2.calcHist([matched_b], [0], None, [256], [0, 256]))
b_norm_matching /= b_norm_matching.sum()    
cdf_b_matching = b_norm_matching.cumsum()

g_norm_matching = (cv2.calcHist([matched_g], [0], None, [256], [0, 256]))
g_norm_matching /= g_norm_matching.sum()    
cdf_g_matching = g_norm_matching.cumsum()

r_norm_matching = (cv2.calcHist([matched_r], [0], None, [256], [0, 256]))
r_norm_matching /= r_norm_matching.sum()    
cdf_r_matching = r_norm_matching.cumsum()



plt.figure(figsize=(10, 6))

# --------------------------------------------- image -------------------------------------------- #
plt.subplot(3, 3, 1)
plt.imshow(image)

plt.subplot(3, 3, 2)
plt.plot(b_norm_image, color='b')
plt.plot(g_norm_image, color='g')
plt.plot(r_norm_image, color='r')

plt.subplot(3, 3, 3)
plt.plot(cdf_b_image, color='b')
plt.plot(cdf_g_image, color='g')
plt.plot(cdf_r_image, color='r')

# ------------------------------------------- template ------------------------------------------- #
plt.subplot(3, 3, 4)
plt.imshow(template)

plt.subplot(3, 3, 5)
plt.plot(b_norm_template, color='b')
plt.plot(g_norm_template, color='g')
plt.plot(r_norm_template, color='r')

plt.subplot(3, 3, 6)
plt.plot(cdf_b_template, color='b')
plt.plot(cdf_g_template, color='g')
plt.plot(cdf_r_template, color='r')


# ---------------------------------------- matching image ---------------------------------------- #

plt.subplot(3, 3, 7)
plt.imshow(matched_image)

plt.subplot(3, 3, 8)
plt.plot(b_norm_matching, color='b')
plt.plot(g_norm_matching, color='g')
plt.plot(r_norm_matching, color='r')

plt.subplot(3, 3, 9)
plt.plot(cdf_b_matching, color='b')
plt.plot(cdf_g_matching, color='g')
plt.plot(cdf_r_matching, color='r')

plt.tight_layout()
plt.show()