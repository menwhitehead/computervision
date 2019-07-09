import numpy as np
import sys
from PIL import Image

# Python3
from pathlib import Path

import edgedetection
import imageutils


filename = Path(sys.argv[1])
output_filename = filename.stem + "_sobeledges" + filename.suffix
img = np.array(Image.open(filename)).astype(np.uint8)
gray = imageutils.grayscale(img)
grads, orientations = edgedetection.sobel(gray)
output = Image.fromarray(grads).convert('L')
output.save(output_filename)










