# Importing libraries
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import cv2 

def denoiser(image, k=5):
    
    rows, cols = image.shape
    image = cv2.copyMakeBorder(image, k//2, k//2, k//2, k//2, cv2.BORDER_REFLECT)
    conv_filter = np.ones((k, k), dtype="uint8") / (k*k)
    
    for i in range(rows):
        for j in range(cols):
            window = image[i : i+ k, j : j + k]
            image[i, j] = np.sum(np.multiply(conv_filter, window))
    
    plt.figure
    plt.imshow(image, cmap = "gray")
    plt.title(f"Image denoised using {k}x{k} filter")
    plt.show()
    
if __name__ == "__main__":
    image = skimage.io.imread("./noisy.tif" )
    filters = [5, 10, 15]
    for k in filters:
        denoiser(image,k)