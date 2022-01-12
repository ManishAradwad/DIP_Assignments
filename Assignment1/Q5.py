"""
Author: ManishAradwad
Date: 25/08/2021

Q5.

Binary Morphology: Binarize the image NoisyImage.png and apply binary morphological
operations to remove the noise in the image.
Function inputs: noisy image
Function outputs: cleaned image.

"""


import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

path = "./NoisyImage.png"
text_image = skimage.io.imread(path)

def remove_noise(image):
    
    thresh = threshold_otsu(image)
    binary_img = image > thresh # Binarizing the image using otsu's algorithm
    rows, cols = binary_img.shape
    final_image = np.zeros((rows, cols), dtype="uint8")
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            
            # Here, a 3x3 window is used. All the intensities of those 9 pixels is stored in temporary array
            temp = [binary_img[i-1, j-1], binary_img[i-1, j], binary_img[i-1, j+1],
                    binary_img[i, j-1], binary_img[i, j], binary_img[i, j+1],
                    binary_img[i+1, j-1], binary_img[i+1, j], binary_img[i+1, j+1]]
            
            # All the intensities are sorted
            temp = sorted(temp)
            
            # The intensity value at middle of sorted array(which in turn is the maximum intensity in the array) is assigned to the current pixel
            final_image[i, j] = temp[4]
    
    plt.figure()
    plt.axis("off")
    plt.imshow(final_image, cmap="gray")
    
if __name__ == "__main__":
    
    remove_noise(text_image)