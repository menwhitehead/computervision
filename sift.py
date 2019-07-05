import math
import numpy as np
import scipy.signal
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Open the image
# orig_img = np.array(Image.open('tutt.jpg')).astype(np.uint8)
# orig_img = np.array(Image.open('magpie.png')).astype(np.uint8)
orig_img = np.array(Image.open('magpie.png')).astype(np.float32)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
# orig_img = cv2.resize(orig_img, (orig_img.shape[1]/2, orig_img.shape[0]/2))
# orig_img = np.array(Image.open('house.png')).astype(np.uint8)


# sift = cv2.SIFT()
# kp = sift.detect(gray, None)
# img2=cv2.drawKeypoints(orig_img, kp)

# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)
# img2=cv2.drawKeypoints(gray,kp)
# cv2.imwrite('sift_keypoints.jpg',img2)
# sys.exit()


# sobelx = cv2.Sobel(orig_img,cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(orig_img,cv2.CV_64F,0,1,ksize=5)
# grads = np.sqrt(sobelx**2 + sobely**2)
# orientations = np.arctan2(sobely, sobelx)

sigma = 1.6
num_octaves = 4
num_scales = 5
k = 1.4

data = []
curr_img = orig_img[:, :]
for octave in range(num_octaves):
    octave_data = []
    for scale in range(num_scales):
        curr_img = cv2.GaussianBlur(curr_img, (5, 5), k * sigma)
        sigma = k * sigma
        octave_data.append(curr_img)
    data.append(octave_data)
    curr_img = cv2.resize(curr_img, (curr_img.shape[1]/2, curr_img.shape[0]/2))


dogs = []
for octave in range(num_octaves):
    octave_data = []
    for scale in range(num_scales-1):
        dog = data[octave][scale] - data[octave][scale+1]
        norm_image = cv2.normalize(dog, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        octave_data.append(norm_image)
        # octave_data.append(dog)
    dogs.append(octave_data)




# print np.array(dogs[0][0]).shape

edge_threshold = 10
contrast_threshold = 150

ext_keypoints = []
for octave in range(num_octaves):
    keypoints = []
    scale_matrix = np.array(dogs[octave])
    # print scale_matrix.shape
    for i in range(1, len(scale_matrix)-1):
        sobelx = cv2.Sobel(scale_matrix[i],cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(scale_matrix[i],cv2.CV_64F,0,1,ksize=3)
        # grads = np.sqrt(sobelx**2 + sobely**2)
        # orientations = np.arctan2(sobely, sobelx)

        output = np.zeros(scale_matrix[0].shape)
        layers = scale_matrix[i-1:i+2][:, :]
        for j in range(1, layers[0].shape[0]-1):
            for k in range(1, layers[0].shape[1]-1):
                chunk = layers[:, j-1:j+2, k-1:k+2]
                current_pixel = chunk[1, 1, 1]

                if (current_pixel > np.max(chunk[0, :, :]) and \
                   current_pixel > np.max(chunk[2, :, :]) and \
                   current_pixel > np.max(chunk[:, 0, :]) and \
                   current_pixel > np.max(chunk[:, 2, :]) and \
                   current_pixel > np.max(chunk[:, :, 0]) and \
                   current_pixel > np.max(chunk[:, :, 2])) or \
                  (current_pixel < np.min(chunk[0, :, :]) and \
                   current_pixel < np.min(chunk[2, :, :]) and \
                   current_pixel < np.min(chunk[:, 0, :]) and \
                   current_pixel < np.min(chunk[:, 2, :]) and \
                   current_pixel < np.min(chunk[:, :, 0]) and \
                   current_pixel < np.min(chunk[:, :, 2])):
                    print "BIGG", i, j, k
                    output[j][k] = 255

                    # if current_pixel > contrast_threshold:
                    #     dx = sobelx[j, k]
                    #     dy = sobely[j, k]
                    #     mat = np.array([[dx**2, dx*dy],[dx*dy, dy**2]])
                    #     lambs, vecs = np.linalg.eig(mat)
                    #     # print lambs[0], lambs[1]
                    #     ratio = lambs[0] / lambs[1]
                    #     # print ratio
                    #     if ratio < edge_threshold and ratio > 1.0/edge_threshold:
                    #         output[j][k] = 255
        keypoints.append(output)
    ext_keypoints.append(keypoints)



ext_keypoints = np.array(ext_keypoints)
print ext_keypoints.shape

oc = 3
sc = 0
pic = dogs[oc][sc]
colorized = cv2.cvtColor(pic, cv2.COLOR_GRAY2BGR)

# for i in range(len(ext_keypoints)):
#     for j in range(len(ext_keypoints[i])):
for k in range(len(ext_keypoints[oc][sc])):
    for m in range(len(ext_keypoints[oc][sc][k])):
        # print ext_keypoints[i][j][k][m], i, j
        if ext_keypoints[oc][sc][k][m] > 250:
            print "REDDY", i, j, k, m
            for x in range(3):
                for y in range(3):
                    colorized[k+x-1][m+y-1] = [0, 0, 255]

# output = Image.fromarray(colorized).convert('RGB')
# output.save("dog_house.jpg")
cv2.imwrite("dog_house.jpg", colorized)

# for octave in range(len(dogs)):
#     for scale in range(len(dogs[octave])):
#         output = Image.fromarray(dogs[octave][scale]).convert('L')
#         output.save("sifted_house%d_%d.jpg" % (octave, scale))





