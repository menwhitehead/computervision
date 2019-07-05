import numpy as np
import sys
from PIL import Image

# Python3
from pathlib import Path

import edgedetection
import imageutils


filename = Path(sys.argv[1])
output_filename = filename.stem + "_laplacianedges" + filename.suffix
img = np.array(Image.open(filename))
img = imageutils.bytesToFloat(img)
gray = imageutils.grayscale(img)
edges = edgedetection.laplacian(gray)
edges = imageutils.normalizeImage(edges)
arr = imageutils.floatToBytes(edges)
output = Image.fromarray(arr)#.convert('L')
output.save(output_filename)










