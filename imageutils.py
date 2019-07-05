import numpy as np
import scipy.signal
import edgedetection



def getGaussianKernel(size=3, sigma=1.4, amp=1.0):
    kernel = np.zeros((size, size))
    center = int(size/2)
    denom = 2 * sigma**2
    for x in range(size):
        x_off = ((x - center) **2) / denom
        for y in range(size):
            y_off = ((y - center) **2) / denom
            kernel[x][y] = amp * np.e**(-(x_off + y_off))
    some = np.sum(kernel)
    kernel /= some

    return kernel


# Blur the image using a Gaussian kernel
def gaussianBlur(img, size, sigma=1.4):
    kernel = getGaussianKernel(size, sigma)
    if img.ndim == 2:
        return scipy.signal.convolve2d(img, kernel, mode="same", boundary="symm").astype(np.uint8)
    elif img.ndim == 3:
        # has RGB 3 channels
        reds = scipy.signal.convolve2d(img[:, :, 0], kernel)
        greens = scipy.signal.convolve2d(img[:, :, 1], kernel)
        blues = scipy.signal.convolve2d(img[:, :, 2], kernel)
        result = np.stack((reds, greens, blues), axis=2)
        return result.astype(np.uint8)


# Convert a color RGB image to grayscale
def grayscale(img):
    if img.ndim == 3:
        gray_img = np.round(0.299 * img[:, :, 0] +
                            0.587 * img[:, :, 1] +
                            0.114 * img[:, :, 2]).astype(np.uint8)
        return gray_img
    else:
        return img


def clipBytes(img):
    return np.clip(img, 0, 255)

def rotateImage(img, radians):
    pass

def cropImage(img, startx, starty, endx, endy):
    pass

def scaleImage(img, ratiox, ratioy):
    pass

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


def drawLine(img, startrow, startcol, endrow, endcol):
    if startcol > endcol:
        startcol, endcol = endcol, startcol
        startrow, endrow = endrow, startrow
    drow = endrow - startrow
    dcol = endcol - startcol
    count = 0
    if dcol > 0:
        for col in range(startcol, endcol):
            row = int(np.round(startrow + drow * (col - startcol) / dcol))
            img[row, col] = 255
            count+=1
    else:
        if startrow > endrow:
            endrow, startrow = startrow, endrow
        for row in range(startrow, endrow):
            img[row, startcol] = 255
            count+=1


def drawLineAngle(img, startrow, startcol, angle, length):
    drow = length * np.sin(angle)
    dcol = length * np.cos(angle)
    endrow = int(np.round(startrow + drow))
    endcol = int(np.round(startcol + dcol))
    drawLine(img, startrow, startcol, endrow, endcol)


# Overlay orientation arrows on a copy of the given image
# Set the resolution to define how dense the grid of arrows is
def orientationImage(img, resolution=25):
    arrow_length = 4
    new_img = img[:]
    grads, orientations = edgedetection.sobel(img)
    chosen = orientations[:-resolution:resolution, :-resolution:resolution]
    chosen_directions = getDirections(chosen)
    for i in range(1, chosen_directions.shape[0]):
        for j in range(1, chosen_directions.shape[1]):
            row = i * resolution
            col = j * resolution
            direction = chosen_directions[i][j]
            drawLineAngle(new_img, row, col, np.pi*(direction/4.), arrow_length)

    return new_img







