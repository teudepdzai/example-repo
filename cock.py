import cv2;
import numpy as np

# Load image
img = cv2.imread("images/grandma.jpeg")

# Upscale 3x using high-quality interpolation
upscaled = cv2.resize(
    img,
    None,
    fx=3,
    fy=3,
    interpolation=cv2.INTER_CUBIC
)

# Sharpening kernel
sharpen_kernel = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
])

# Apply sharpening
sharpened = cv2.filter2D(upscaled, -1, sharpen_kernel)

# Save result
cv2.imwrite("output_x3_sharpened.jpg", sharpened)
