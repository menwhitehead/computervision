import numpy as np
import sys
from PIL import Image

# Python3
from pathlib import Path

import edgedetection
import imageutils


filename = Path(sys.argv[1])
output_filename = filename.stem + "_cannyedges" + filename.suffix
img = np.array(Image.open(filename)).astype(np.uint8)
gray = imageutils.grayscale(img)
edges = edgedetection.canny(gray)
output = Image.fromarray(edges).convert('L')
output.save("generated_images/" + output_filename)










