import numpy as np
import scipy.signal
import edgedetection

def getGaussianKernel(size=3):
    gaussian_filter = (1/273.) * np.array([[ 1.0278445 ,  4.10018648,  6.49510362,  4.10018648,  1.0278445 ],
               [ 4.10018648, 16.35610171, 25.90969361, 16.35610171,  4.10018648],
               [ 6.49510362, 25.90969361, 41.0435344 , 25.90969361,  6.49510362],
               [ 4.10018648, 16.35610171, 25.90969361, 16.35610171,  4.10018648],
               [ 1.0278445 ,  4.10018648,  6.49510362,  4.10018648,  1.0278445 ]])
    return gaussian_filter

# Blur the image using a Gaussian kernel
def gaussianBlur(img, size):
    kernel = getGaussianKernel(size)
    return scipy.signal.convolve2d(img, kernel)


# Convert a color RGB image to grayscale
def grayscale(img):
    gray_img = np.round(0.299 * img[:, :, 0] +
                        0.587 * img[:, :, 1] +
                        0.114 * img[:, :, 2]).astype(np.uint8)
    return gray_img


def clipBytes(img):
    return np.clip(img, 0, 255)


def normalizeImage(img):
    small = np.min(img)
    large = np.max(img)
    return ((img - small) / (large - small)) * 255

def getDirections(orientations, number_bins=8):
    # Orientations are values between pi and -pi
    # Output directions are:
    #   0 - East
    #   1 - Northeast
    #   2 - North
    #   3 - Northwest
    #   4 - West
    #   5 - Southwest
    #   6 - South
    #   7 - Southeast
    directions = ((orientations + np.pi) / (2*np.pi)) * number_bins
    directions = directions.round()
    directions = directions.astype(int)
    directions = (directions + 4) % number_bins
    return directions

# Overlay orientation arrows on a copy of the given image
# Set the resolution to define how dense the grid of arrows is
def orientationImage(img, resolution=25):
    arrow_length = 4
    new_img = img[:]
    grads, orientations = edgedetection.sobel(img)
    chosen = orientations[resolution:-resolution:resolution, resolution:-resolution:resolution]
    chosen_directions = getDirections(chosen)
    for i in range(chosen_directions.shape[0]):
        for j in range(chosen_directions.shape[1]):
            row = i * resolution
            col = j * resolution
            direction = chosen_directions[i][j]

            if direction == 0:
                new_img[row, col:col+arrow_length] = 255
            elif direction == 1:
                new_img[row, col] = 255
                new_img[row-1, col+1] = 255
                new_img[row-2, col+2] = 255
            elif direction == 2:
                new_img[row-arrow_length:row, col] = 255
            elif direction == 3:
                new_img[row, col] = 255
                new_img[row-1, col-1] = 255
                new_img[row-2, col-2] = 255
            elif direction == 4:
                new_img[row, col-arrow_length:col] = 255
            elif direction == 5:
                new_img[row, col] = 255
                new_img[row+1, col-1] = 255
                new_img[row+2, col-2] = 255
            elif direction == 6:
                new_img[row:row+arrow_length, col] = 255
            elif direction == 7:
                new_img[row, col] = 255
                new_img[row+1, col+1] = 255
                new_img[row+2, col+2] = 255


    return new_img







