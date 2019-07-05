import numpy as np
import scipy.signal
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Open the image
# img = np.array(Image.open('chimney1.png')).astype(np.uint8)
# img = np.array(Image.open('theo.jpg')).astype(np.uint8)
# img = np.array(Image.open('/home/mwhitehead/Desktop/girl_flower.png')).astype(np.uint8)
img = np.array(Image.open('tutt.jpg')).astype(np.uint8)
orig_img = np.array(Image.open('tutt.jpg'))

# Apply gray scale
gray_img = np.round(0.299 * img[:, :, 0] +
                    0.587 * img[:, :, 1] +
                    0.114 * img[:, :, 2]).astype(np.uint8)

# define filters
horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1


horizontal_grads = scipy.signal.convolve2d(gray_img, horizontal, mode="same")
vertical_grads = scipy.signal.convolve2d(gray_img, vertical, mode="same")

# gradients = np.sqrt(pow(horizontal_grads, 2.0) + pow(vertical_grads, 2.0))
# directions = np.arctan2(vertical_grads, horizontal_grads)
# directions = ((directions + np.pi) / (2*np.pi)) * 8
# directions = directions.round()
# directions = directions.astype(int)
# directions = directions % 4
h, w = gray_img.shape
threshold = 10000

corners = []
for row in range(1, h-1):
    for col in range(1, w-1):
        gx = horizontal_grads[row-1:row+2, col-1:col+2]
        gy = vertical_grads[row-1:row+2, col-1:col+2]
        gx_squared = np.sum(gx ** 2)
        gy_squared = np.sum(gy ** 2)
        gx_gy = np.sum(gx * gy)
        # print "NEXT", gx_squared, gy_squared, gx_gy
        m = np.array([[gx_squared, gx_gy],[gx_gy, gy_squared]])
        lamb1, lamb2 = np.linalg.eig(m)[0]
        #r = (lamb1 * lamb2) / (lamb1 + lamb2)
        r = (lamb1 * lamb2) - 0.05 * (lamb1 + lamb2)**2
        corners.append((r, row, col))

corners.sort()
k = 10000
for r, row, col in corners[-k:]:
    orig_img[row, col] = (255, 0, 0)


# corners_img = cv2.cornerHarris(gray_img, 3, 3, 0.04)
#
# #Marking the corners in Green
# orig_img[corners_img>0.001*corners_img.max()] = [255,0,0]

# final_grads = np.where(both_grads > 0, 255.0, 0.0)
output = Image.fromarray(orig_img)
output.save("corners_harris.jpg")

