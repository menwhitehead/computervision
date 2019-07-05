import math
import numpy as np
import scipy.signal
from PIL import Image
import cv2

import imageutils
from pathlib import PurePath



def createScaleSpace(orig_img, num_octaves, num_scales, k, sigma=1.4):
    ssdata = []
    curr_img = orig_img[:, :]
    print("CURR:", curr_img.dtype)
    for octave in range(num_octaves):
        octave_data = []
        for scale in range(num_scales):
            #curr_img = cv2.GaussianBlur(curr_img, (5, 5), k * sigma)
            curr_img = imageutils.gaussianBlur(curr_img, 3, k * sigma)
            print("CURR:", curr_img.dtype)
            sigma = k * sigma
            octave_data.append(curr_img)
        ssdata.append(octave_data)
        curr_img = cv2.resize(curr_img, (int(curr_img.shape[1]/2), int(curr_img.shape[0]/2)))
    return ssdata

def dumpSS(ssdata, num_octaves, num_scales):
    folder = PurePath("tmp")
    for octave in range(num_octaves):
        for scale in range(num_scales):
            arr = imageutils.floatToBytes(ssdata[octave][scale])

            output = Image.fromarray(arr).convert("L")
            out = folder / ("ss_%d_%d.jpg" % (octave, scale))
            # print(out)
            output.save(str(out))

def dumpDogs(dogs, num_octaves, num_scales):
    folder = PurePath("tmp")
    for octave in range(num_octaves):
        for scale in range(num_scales-1):
            arr = imageutils.floatToBytes(dogs[octave][scale])
            output = Image.fromarray(arr).convert("L")
            out = folder / ("dog_%d_%d.jpg" % (octave, scale))
            # print(out)
            output.save(str(out))

def createDiffGaussians(ssdata, num_octaves, num_scales):
    dogs = []
    for octave in range(num_octaves):
        octave_data = []
        for scale in range(num_scales-1):
            dog = ssdata[octave][scale] - ssdata[octave][scale+1]
            # norm_image = cv2.normalize(dog, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # print(dog.dtype)
            norm_image = imageutils.normalizeImage(dog)
            # print("MAX:", np.max(norm_image))
            octave_data.append(norm_image)
            # octave_data.append(dog)
        dogs.append(octave_data)
    return dogs

def findExtrema(dogs, num_octaves, num_scales):
    ext_keypoints = []
    for octave in range(num_octaves):
        keypoints = []
        scale_matrix = np.array(dogs[octave])
        # print(scale_matrix[0])
        for i in range(1, len(scale_matrix)-1):
            # sobelx, sobely = edgedetection.getSobelGradients(scale_matrix[i])

            output = np.zeros(scale_matrix[0].shape)
            layers = scale_matrix[i-1:i+2][:, :]
            for j in range(1, layers[0].shape[0]-1):
                for k in range(1, layers[0].shape[1]-1):
                    chunk = layers[:, j-1:j+2, k-1:k+2]
                    current_pixel = chunk[1, 1, 1]
                    reduction = chunk - (current_pixel + .001)
                    # print(chunk)
                    # print()
                    # print(reduction)
                    # sys.exit()
                    # if np.all(reduction < 0):
                    #     output[j][k] = 255

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
                        # print "BIGG", i, j, k
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
    return ext_keypoints


def thresholdKeypoints(dogs, ext_keypoints, threshold=250):
    oc = 0
    sc = 0
    pic = dogs[oc][sc]
    final_keypoints = []
    # colorized = cv2.cvtColor(pic, cv2.COLOR_GRAY2BGR)


    for k in range(len(ext_keypoints[oc][sc])):
        for m in range(len(ext_keypoints[oc][sc][k])):
            if ext_keypoints[oc][sc][k][m] > threshold:
                final_keypoints.append((k, m))
                # for x in range(3):
                #     for y in range(3):
                #         colorized[k+x-1][m+y-1] = [0, 0, 255]

    return final_keypoints

def sift(img):
    sigma = 1.1
    num_octaves = 4
    num_scales = 5
    k = 1.4

    gray = imageutils.grayscale(img)
    normalized = imageutils.bytesToFloat(gray)
    print("NORMALIZED:", normalized.dtype)

    ssdata = createScaleSpace(normalized, num_octaves, num_scales, k, sigma)
    dumpSS(ssdata, num_octaves, num_scales)

    dogs = createDiffGaussians(ssdata, num_octaves, num_scales)
    dumpDogs(dogs, num_octaves, num_scales)

    extrema = findExtrema(dogs, num_octaves, num_scales)
    pic = thresholdKeypoints(dogs, extrema)
    # print(extrema[0][0])
    return pic
    # edge_threshold = 10
    # contrast_threshold = 150






