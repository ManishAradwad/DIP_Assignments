# Importing libraries
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np

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

# 5x5 gaussian filter
def GaussianFilter(image, sigma=5, k=5):

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

# Edge detector using Prewitt Kernel
def edge_detector_prew(image, t):
    
    m, n = image.shape
    prewitt_op_x = np.zeros([3, 3])
    prewitt_op_y = np.zeros([3, 3])
    prewitt_op_x[0, :] = -1
    prewitt_op_x[2, :] = 1
    prewitt_op_y[:, 0] = -1
    prewitt_op_y[:, 2] = 1
    
    img_grad_x = convolution(image, prewitt_op_x, 3)
    img_grad_y = convolution(image, prewitt_op_y, 3)
    
    edged = np.zeros([m, n])
    
    for i in range(m):
        for j in range(n):
            temp = np.sqrt(img_grad_x[i, j]**2 + img_grad_y[i, j]**2)
            if temp > t:
                edged[i, j] = 1
            else:
                edged[i, j] = 0

    return edged

# Edge detector using Laplacian Kernel
def edge_detector_lapl(image, t, noisy=False):
    
    m, n = image.shape
    lapl_op = np.zeros([3, 3])
    lapl_op[1, 1], lapl_op[0, 1], lapl_op[2, 1], lapl_op[1, 0], lapl_op[1, 2] = -4, 1, 1, 1, 1
    
    convolved = convolution(image, lapl_op, 3)
    
    if noisy == True:
        convolved[convolved > t] = 1
    
    edged = zc_detection(convolved)
    return edged

# Function for zero cross detection
def zc_detection(image):
    m, n = image.shape
    zc_image = np.zeros([m, n])
    
    for i in range(m-1):
        for j in range(n-1):
            if image[i, j] > 0:
                if image[i+1, j] < 0 or image[i, j+1] < 0 or image[i+1, j+1] < 0:
                    zc_image[i, j] = 1
            elif image[i, j] < 0:
                if image[i+1, j] > 0 or image[i, j+1] > 0 or image[i+1, j+1] > 0:
                    zc_image[i, j] = 1
            elif image[i, j] == 0:
                if image[i+1, j+1] != 0 or image[i, j+1] != 0 or image[i+1, j] != 0:
                    zc_image[i, j] = 1

    return zc_image

if __name__ == "__main__":
    
    images = ["Checkerboard.png", "NoisyCheckerboard.png", "Coins.png", "NoisyCoins.png"]
    
    for image in images:
        
        # Original Image loaded
        print(f"Processing {image}: ")
        image1 = skimage.io.imread("./" + image)
        
        fig = plt.figure()
        fig.suptitle(f"{image}")
        
        plt.subplot(2, 2, 1)
        plt.axis("off")
        plt.imshow(image1, cmap="gray")
        plt.title("Original")
        
        # Applying gaussian filter
        print("Convolving with Gaussian Filter")
        image1 = GaussianFilter(image1)
        plt.subplot(2, 2, 2)
        plt.axis("off")
        plt.imshow(image1, cmap="gray")
        plt.title("a) Gaussian Filtered")
        
        # Edge detection using Prewitt operator
        print("Finding edges with Prewitt operator")
        image1 = edge_detector_prew(image1, 10)
        plt.subplot(2, 2, 3)
        plt.axis("off")
        plt.imshow(image1, cmap="gray")
        plt.title("b) Prewitt Edges")
        
        # Edge detection using Laplacian operator
        print("Finding edges with Laplacian operator\n\n")
        if image in ["NoisyCheckerboard.png", "NoisyCoins.png"]:
            image1 = edge_detector_lapl(image1, 10, True)
        else:
            image1 = edge_detector_lapl(image1, 10)
        
        plt.subplot(2, 2, 4)
        plt.axis("off")
        plt.imshow(image1, cmap="gray")
        plt.title("c) Laplacian Edges")
        
        plt.show()