import numpy as np
import scipy.signal
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Open the image
img = np.array(Image.open('house.png')).astype(np.uint8)
edges = cv2.Canny(img, 100, 200)

for row in range(edges.shape[0]):
    for col in range(edges.shape[1]):
        if edges[row][col] > 0:
            print edges[row][col], row, col




output = Image.fromarray(edges).convert('L')
output.save("edges_canny.jpg")
