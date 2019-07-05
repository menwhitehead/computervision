import numpy as np
import sys
from PIL import Image

# Python3
from pathlib import Path

import edgedetection
import imageutils


filename = Path(sys.argv[1])
output_filename = filename.stem + "_gaussian" + filename.suffix
img = Image.open(filename)
kind = img.mode
arr = np.array(img).astype(np.uint8)
blurred = imageutils.gaussianBlur(arr, 7, 1.04)
# blurred = imageutils.normalizeImage(blurred)
# print(blurred)
print(kind)
output = Image.fromarray(blurred).convert(kind)
output.save(output_filename)










