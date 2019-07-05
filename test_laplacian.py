import numpy as np
import sys
from PIL import Image

# Python3
from pathlib import Path

import edgedetection
import imageutils


filename = Path(sys.argv[1])
output_filename = filename.stem + "_laplacianedges" + filename.suffix
img = np.array(Image.open(filename)).astype(np.uint8)
gray = imageutils.grayscale(img)
edges = edgedetection.laplacian(gray)
print(edges)
edges = imageutils.normalizeImage(edges)
print(edges)
output = Image.fromarray(edges).convert('L')
output.save(output_filename)










