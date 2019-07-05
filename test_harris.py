import numpy as np
import sys
from PIL import Image

# Python3
from pathlib import Path

import cornerdetection
import imageutils

filename = Path(sys.argv[1])
output_filename = filename.stem + "_harris" + filename.suffix
img = Image.open(filename)
kind = img.mode
arr = np.array(img).astype(np.uint8)
corners = cornerdetection.harris(arr)

# Autodetect a reasonable threshold
max_corners = np.max(corners)
threshold = 0.01 * max_corners

# Mark found corners in RED
reds = np.where(corners > threshold, 255, arr[:, :, 0])
arr[:, :, 0] = reds

output = Image.fromarray(arr).convert(kind)
output.save(output_filename)










