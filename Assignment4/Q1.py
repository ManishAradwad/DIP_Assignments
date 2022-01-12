# Importing libraries
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Function to downsample the filtered and unfiltered image based on input
def down_sample(image, type_img):
    
    if type_img == "uf":
        down_img = image[::2, ::2]
        plt.figure()
        plt.imshow(down_img, cmap="gray")
        plt.title("Downsampling unfiltered Image")
        plt.show()
        
    elif type_img == "f":  
        # Trying different values of window size and sigma
        for k in [3, 5]:
            for sigma in [1, 10]:
                image = GaussianFilter(image, sigma, k)
                down_img = image[::2, ::2]
                plt.figure()
                plt.imshow(down_img, cmap="gray")
                plt.title(f"Downsampling Gaussian filtered Image: \n{k}x{k} Window and {sigma} Sigma")
                plt.show()

# Function to perform convolution
def convolution(image, kernel, k):
    
    m, n = image.shape
    pad_image = np.pad(image, pad_width=((k//2, k//2), (k//2, k//2)), mode="constant",
                       constant_values=0).astype(np.float32)
    
    img_conv = np.zeros([m, n])
    l = k//2
    
    for i in range(l, m-l):
        for j in range(l, n-l):
            x = pad_image[i-l:i-l+k, j-l:j-l+k]
            x = x.flatten() * kernel.flatten()
            img_conv[i, j] = x.sum()
    
    return img_conv

# kxk gaussian filter
def GaussianFilter(image, sigma=1, k=5):

    gaussian_kernel = np.zeros([k, k])
    
    l = k//2
    for x in range(-l, l+1):
        for y in range(-l, l+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2 * sigma**2))
            gaussian_kernel[x+l, y+l] = (1/x1)*x2
            
    filtered_img = np.zeros(image.shape)
    filtered_img = convolution(image, gaussian_kernel, k)
    
    return filtered_img

if __name__ == "__main__":
    
    image = skimage.io.imread("./barbara.tif")
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    
    down_sample(image, 'uf')
    down_sample(image, 'f')
        
    lib_img = cv2.pyrDown(image)
    plt.figure()
    plt.imshow(lib_img, cmap="gray")
    plt.title("Downsampling unfiltered Image using OpenCV")
    plt.show()