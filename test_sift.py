import numpy as np
import sys
from PIL import Image

# Python3
from pathlib import Path

import featuredetection
import imageutils

filename = Path(sys.argv[1])
output_filename = filename.stem + "_sift" + filename.suffix
img = Image.open(filename)
kind = img.mode
arr = np.array(img).astype(np.uint8)
features = featuredetection.sift(arr)
# print(features)
# for (x, y) in features:
#     arr[x, y] = (255, 0, 0)
# output = Image.fromarray(arr).convert(kind)
# output.save(output_filename)










