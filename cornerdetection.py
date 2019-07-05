import numpy as np
import scipy.signal
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import edgedetection
import imageutils


def harris(img, k=0.05):
    k = 0.05

    gray = imageutils.grayscale(img)
    h, w = gray.shape
    horizontal_grads, vertical_grads = edgedetection.getSobelGradients(gray)

    gx_squareds = horizontal_grads**2
    gx_squared_sums = scipy.signal.convolve2d(gx_squareds, np.ones((3,3)), mode="same", boundary="symm")

    gy_squareds = vertical_grads**2
    gy_squared_sums = scipy.signal.convolve2d(gy_squareds, np.ones((3,3)), mode="same", boundary="symm")

    gx_gys = horizontal_grads * vertical_grads
    gx_gy_sums = scipy.signal.convolve2d(gx_gys, np.ones((3,3)), mode="same", boundary="symm")

    deters = gx_squared_sums * gy_squared_sums - gx_gy_sums**2
    traces = (gx_squared_sums + gy_squared_sums) ** 2
    corners = deters - k * traces

    return corners



