# Importing libraries
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import ndimage as ndi
import cv2
from PIL import Image
from skimage.util import random_noise

def grad_x(image):
    kernel = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return signal.convolve2d(image, kernel, mode="same")

def grad_y(image):
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return signal.convolve2d(image, kernel, mode="same")

def harris_detector(image, t = 1e8):
    I_x = grad_x(image)
    I_y = grad_y(image)
    
    I_xx = ndi.gaussian_filter(I_x**2, sigma=1)
    I_xy = ndi.gaussian_filter(I_y*I_x, sigma=1)
    I_yy = ndi.gaussian_filter(I_y**2, sigma=1)
    
    k=0.05
    detM = I_xx * I_yy - I_xy**2
    traceM = I_xx + I_yy
    R = detM - k * (traceM ** 2)
    
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    for rowIndex, response in enumerate(R):
            for colIndex, r in enumerate(response):
                if r > t:
                    result[rowIndex, colIndex] = [255, 0, 0]
                
    return result

def plotter(orig_img, corner_img, img_type):
    
    fig = plt.figure()
    fig.suptitle(f"{img_type} {image}")
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(orig_img, cmap="gray")
    plt.title("Original")
    
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.figure()
    plt.imshow(corner_img)
    plt.title("Detected Corners")
    plt.show()

if __name__ == "__main__":

    images = ["Checkerboard.png", "MainBuilding.png"]
        
    for image in images:
        
        # Original Image loaded
        image1 = skimage.io.imread("./" + image)
        
        # Part a)
        for t in [1e7, 1e8]:
            
            fig = plt.figure()
            fig.suptitle(f"Normal {image}")
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.imshow(image1, cmap="gray")
            plt.title("Original")
        
            img_corner = harris_detector(image1, t)
            
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.imshow(img_corner)
            plt.title(f"Detected Corners\n(t={t})")
        
        # Part b)
        img = Image.open("./" + image)
        
        image1_rot = np.array(img.rotate(60))
        img_rot_corner = harris_detector(image1_rot)
        plotter(image1_rot, img_rot_corner, "Rotated")
        
        m, n = img.size
        img_scaled = np.array(img.crop((4, n/5, 1200, 4/5 * m)).resize((300, 300)))
        img_sc_corner = harris_detector(img_scaled)
        plotter(img_scaled, img_sc_corner, "Scaled")
        
        noise_img = random_noise(np.array(img), mode="gaussian", var=0.08**2)
        noise_img = (255*noise_img).astype(np.uint8)
        img_ns_corner = harris_detector(noise_img)
        plotter(noise_img, img_ns_corner, "Noisy")
        