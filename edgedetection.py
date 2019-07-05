import numpy as np
import scipy.signal

import imageutils

# Sobel edge detection
#
# Calculate pixel gradients and orientations from a given numpy array
# with 0-255 grayscale values
# Return a gradient matrix and an orientation matrix
def sobel(img):
    horizontal_grads, vertical_grads = getSobelGradients(img)
    img_grads = np.sqrt(pow(horizontal_grads, 2.0) + pow(vertical_grads, 2.0))
    img_orientations = np.arctan2(vertical_grads, horizontal_grads)
    return img_grads, img_orientations


def getSobelGradients(img):
    # Define filters
    horizontal_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    vertical_kernel = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    horizontal_grads = scipy.signal.convolve2d(img, horizontal_kernel, mode="same", boundary="symm")
    vertical_grads = scipy.signal.convolve2d(img, vertical_kernel, mode="same", boundary="symm")

    return horizontal_grads, vertical_grads


# Laplacian edge detection

def laplacian(img):
    # Define filter
    lap_kernel = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]])

    return scipy.signal.convolve2d(img, lap_kernel)




def canny(img):
    gray_img = imageutils.gaussianBlur(img, 5)
    gradients, directions = sobel(gray_img)

    # Bin the orientations into 4 general directions
    directions = ((directions + np.pi) / (2*np.pi)) * 8
    directions = directions.round()
    directions = directions.astype(int)
    directions = directions % 4


    # horizontals = np.where(directions == 0, gradients, 0)
    # uprights = np.where(directions == 1, gradients, 0)
    # verticals = np.where(directions == 2, gradients, 0)
    # uplefts = np.where(directions == 3, gradients, 0)


    new_grads = np.zeros(gradients.shape)
    for row in range(1, len(gradients)-1):
        for col in range(1, len(gradients[row])-1):
            # vertical angle (horizontal edge)
            if directions[row][col] == 2:
                if gradients[row][col] > gradients[row-1][col] and gradients[row][col] > gradients[row+1][col]:
                    new_grads[row][col] = gradients[row][col]

            # horizontal angle (vertical edge)
            elif directions[row][col] == 0:
                if gradients[row][col] > gradients[row][col-1] and gradients[row][col] > gradients[row][col+1]:
                    new_grads[row][col] = gradients[row][col]

            # up-right diagonal
            elif directions[row][col] == 1:
                if gradients[row][col] > gradients[row-1][col-1] and gradients[row][col] > gradients[row+1][col+1]:
                    new_grads[row][col] = gradients[row][col]

            # down-right diagonal
            elif directions[row][col] == 3:
                if gradients[row][col] > gradients[row+1][col-1] and gradients[row][col] > gradients[row-1][col+1]:
                    new_grads[row][col] = gradients[row][col]

    # Threshold by finding the image median and then setting high/low thresholds
    # a little above and below that median
    middle = np.median(img)
    sigma = 0.33
    high_threshold = int(min(255, (1.0 + sigma) * middle))
    low_threshold = int(max(0, (1.0 - sigma) * middle))

    high_grads = np.where(new_grads > high_threshold, new_grads, 0)
    low_grads = np.where(new_grads > low_threshold, new_grads, 0)

    x = scipy.signal.convolve2d(high_grads, np.ones((3,3)), mode="same", boundary="symm")
    upped_grads = np.where(low_grads > x, low_grads, 0)

    both_grads = high_grads + upped_grads
    final_grads = np.where(both_grads > 0, 255.0, 0.0)

    return final_grads










