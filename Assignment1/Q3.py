"""
Author: ManishAradwad
Date: 25/08/2021

Q3.

Foreground Extraction: For the image SingleColorText_Gray.png, separate the foreground
text from the background using otsu binarization. Display the text in red color on the green
background in GrassBackground.png.
Function inputs: text and background images
Function outputs: an image with the text in red color superimposed on the background.

"""


import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

path = "./SingleColorText_Gray.png"
bgPath = "./GrassBackground.png"
text_image = skimage.io.imread(path)
bgImage = skimage.io.imread(bgPath)

def foreground_extraction(image, bgImage):
        
    # Finding optimum threshold value using otsu's algorithm 
    thresh = threshold_otsu(image)
    
    # Getting binary image using above threshold. Note that this binary image has intensity <= threshold as 1 and > threshold as 0
    # That means the text will be 0 and rest is 1
    binary_img = image <= thresh
    
    finalImage = np.zeros((image.shape[0], image.shape[1], 3), dtype = 'uint8')
    
    # ORing the original image's R channel with binary_image(which adds the text in red only) 
    # and ANDing it with G, B channel(which removes the pixels in text area leaving behind red text only)
    finalImage[:, :, 0] = bgImage[:, :, 0] + binary_img
    finalImage[:, :, 1] = bgImage[:, :, 1] * binary_img
    finalImage[:, :, 2] = bgImage[:, :, 2] * binary_img
    
    plt.figure()
    plt.axis("off")
    plt.imshow(finalImage, cmap="gray")

if __name__ == "__main__":
    foreground_extraction(text_image, bgImage)