"""
Author: ManishAradwad
Date: 23/09/2021

Q2.

"""

# Importing libraries
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import Counter
import math  

# Function to downsample the image
def subsampler(image, k):
    
    image = np.array(image, dtype="uint32")
    
    rows, cols = image.shape
    sub_rows, sub_cols = math.floor(rows / k), math.ceil(cols / k)
    
    op_img = np.zeros((sub_rows, sub_cols), dtype="uint8")
    
    for i in range(sub_rows):
        for j in range(sub_cols):
            op_img[i, j] = image[k*i, k*j]
    
    return op_img
    
# Function which upsamples the image by factor k using both Nearest Neighbor and Bilinear Interpolation
def upsample(image, k):
    rows, cols = image.shape
    up_rows, up_cols = rows * k, cols * k
    
    # Nearest Neighbor Interpolation
    row_ratio = up_rows / rows
    col_ratio = up_cols / cols
    new_row_positions = np.array(range(up_rows))+1
    new_col_positions = np.array(range(up_cols))+1
    
    # normalize new row and col positions by ratios
    new_row_positions = new_row_positions / row_ratio
    new_col_positions = new_col_positions / col_ratio
    
    # apply ceil to normalized new row and col positions
    new_row_positions = np.ceil(new_row_positions)
    new_col_positions = np.ceil(new_col_positions)
    
    # find how many times to repeat each element
    row_repeats = np.array(list(Counter(new_row_positions).values()))
    col_repeats = np.array(list(Counter(new_col_positions).values()))
    
    # perform column-wise interpolation on the columns of the matrix
    row_matrix = np.dstack([np.repeat(image[:, i], row_repeats) 
                            for i in range(cols)])[0]
    
    # perform column-wise interpolation on the columns of the matrix
    nrow, ncol = row_matrix.shape
    nni_img = np.stack([np.repeat(row_matrix[i, :], col_repeats)
                              for i in range(nrow)])
    
    
    # Bilinear Interpolation
    enlargedShape = list(map(int, [image.shape[0]*k, image.shape[1]*k]))
    bi_img = np.zeros(enlargedShape, dtype=np.uint8)
    rowScale = float(image.shape[0]) / float(bi_img.shape[0])
    colScale = float(image.shape[1]) / float(bi_img.shape[1])
 
    for r in range(bi_img.shape[0]):
        for c in range(bi_img.shape[1]):
            orir = r * rowScale # Find position in original image
            oric = c * colScale
            bi_img[r, c] = GetBilinearPixel(image, oric, orir)
 
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(f"Subsampled Image by factor {k}")
    plt.savefig(f"Subsampled Image by factor {k}.png")
    plt.figure()
    plt.imshow(nni_img, cmap="gray")
    plt.title(f"Nearest Neighbour Interpolated Image by factor {k}")
    plt.savefig(f"Nearest Neighbour Interpolated Image by factor {k}.png")
    plt.figure()
    plt.imshow(bi_img, cmap="gray")
    plt.title(f"Bilinear Interpolated Image by factor {k}")
    plt.savefig(f"Bilinear Interpolated Image by factor {k}.png")
    plt.show()
    
    return nni_img, bi_img

# Function to find the value of pixel while interpolation
def GetBilinearPixel(imArr, posX, posY):
    modXi = int(posX)
    modYi = int(posY)
    modXf = posX - modXi
    modYf = posY - modYi
    modXiPlusOneLim = min(modXi+1, imArr.shape[1]-1)
    modYiPlusOneLim = min(modYi+1, imArr.shape[0]-1)
    
    
    bl = imArr[modYi, modXi]
    br = imArr[modYi, modXiPlusOneLim]
    tl = imArr[modYiPlusOneLim, modYi]
    tl = imArr[modYiPlusOneLim, modXi]
    tr = imArr[modYiPlusOneLim, modXiPlusOneLim]
    
    b = modXf * br + (1. - modXf) * bl
    t = modXf * tr + (1. - modXf) * tl
    pxf = modYf * t + (1. - modYf) * b
    out = int(pxf+0.5)
    
    return out
    
if __name__ == "__main__":
    
    inputs = [
               "Bee.jpg", 
               "StoneFace.png"
             ]
    
    for ip in inputs:
        for k in range(2, 4):
            if ip == "Bee.jpg":
                image = Image.open("./" + ip).convert('L')
            else:
                image = skimage.io.imread("./" + ip)
                
            image = np.array(image)
            plt.imshow(image, cmap = "gray")
            plt.title("Original Image")
            subsampled_ip = subsampler(image, k)
            nni_img, bi_img = upsample(subsampled_ip, k)
            
            if (nni_img.shape == image.shape):
                mean_sq_error_nni = np.square(np.subtract(nni_img, image)).mean()
                print(f"Mean Squared Error for {ip} using Nearest Neighbor Interpolation by factor {k} = {mean_sq_error_nni}" )
                mean_sq_error_bi = np.square(np.subtract(bi_img, image)).mean()
                print(f"Mean Squared Error for {ip} using Bilinear Interpolation by factor {k} = {mean_sq_error_bi}\n")
            else:
                mean_sq_error_nni = np.square(np.subtract(nni_img, image[:-1][:])).mean()
                print(f"Mean Squared Error for {ip} using Nearest Neighbor Interpolation by factor {k} = {mean_sq_error_nni}")
                mean_sq_error_bi = np.square(np.subtract(bi_img, image[:-1][:])).mean()
                print(f"Mean Squared Error for {ip} using Bilinear Interpolation by factor {k} = {mean_sq_error_bi}\n")
                