# Incomplete Answer: Not being able to count connected components(digits)

"""
Author: ManishAradwad
Date: 25/08/2021

Q4.

Connected Components: Binarize the image PiNumbers.png and count the number of digits
(0 − 9) using connected component analysis. Also compute the number of occurrences of the digit
1.
Function inputs: PiNumbers.png image
Function outputs: total number of digits, number of occurences of digit 1.

"""

import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

path = "./PiNumbers.png"
text_image = skimage.io.imread(path)

def connected_components(image):
        
    thresh = threshold_otsu(image)
    binary_img = image > thresh
    rows, cols = binary_img.shape
    region_idx = np.zeros((rows, cols), dtype="uint8")
    # region_idx = binary_img.copy()
    region_counter = 1
    # equivalency_list = []
    
    for i in range(rows):
        for j in range(cols):
            if (binary_img[i][j] == 1):
                
                if (binary_img[i][j-1] == 0 and binary_img[i-1][j] == 0):
                    region_idx[i][j] = region_counter
                    region_counter += 1
                
                elif (binary_img[i][j-1] == 0 and binary_img[i-1][j] == 1):
                    region_idx[i][j] = region_idx[i-1][j]
                
                elif (binary_img[i][j-1] == 1 and binary_img[i-1][j] == 0):
                    region_idx[i][j] = region_idx[i][j-1]
                
                elif (binary_img[i][j-1] == 1 and binary_img[i-1][j] == 1):
                
                    if region_idx[i-1][j] == region_idx[i][j-1]:
                        region_idx[i][j] = region_idx[i][j-1]
                    
                    else:
                        # equivalency_list.append([max(region_idx[i][j-1], region_idx[i-1][j]), min(region_idx[i][j-1], region_idx[i-1][j])])
                        min_int = min(region_idx[i][j-1], region_idx[i-1][j])
                        max_int = max(region_idx[i][j-1], region_idx[i-1][j])
                        region_idx[i][j] = min_int
                        region_idx[region_idx == max_int] = min_int
                        # values = np.unique(region_idx)
    
    # equivalency_list.reverse()
    # print(equivalency_list)
    # for i in equivalency_list:
    #     region_idx[region_idx == i[0]] = i[1]
            
    values = np.unique(region_idx)
    
    print("Number of digits in the image: ", len(values))
    
    plt.figure()
    plt.imshow(region_idx, cmap="gray")

connected_components(text_image)