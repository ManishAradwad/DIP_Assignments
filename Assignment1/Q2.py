"""
Author: ManishAradwad
Date: 25/08/2021

Q2.

Otsu’s Binarization: In the class, we showed that σ w
(t) + σ b 2 (t) = σ T 2 (t), where t is the threshold
for binarization. Compute the threshold t for the image coins.png by:
2
.
(a) Minimizing the within class variance σ w
(b) Maximizing the between class variance σ b 2 .
Verify that both methods are equivalent. Compare the time taken by each of the approach and
also compare with the library function.
Function inputs: grayscale image
Function outputs: thresholds from both approaches, time taken by both approaches,
binarized image and threshold from the library function.

"""

import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
import time
np.seterr(divide='ignore', invalid='ignore') # To suppress the unwanted warnings

path = "./Coins.png"
image = skimage.io.imread(path)

# Applying otsu's algorithm using skimage's threshold_otsu function
def otsu_library(image):
    
    thresh = threshold_otsu(image)
    binary_img = image > thresh
    plt.figure()
    plt.axis("off")
    plt.imshow(binary_img, cmap="gray")
    plt.title("Output using skimage's threshold_otsu function")
    print("\nThreshold found with skimage's threshold_otsu function: ", thresh)

# Calculating the threshold by maximizing the between class variance(sigma_b)
def max_between_classv(gray):
    
    total_pixels = gray.shape[0] * gray.shape[1]
    mean_weight = 1.0/total_pixels 
    
    his, bins = np.histogram(gray, np.arange(0,257)) # Finding count of each intensity value
    final_thresh = -1
    final_sigma = -1 # Assigning minimum value to final_sigma since it will be maximized
    intensity_arr = np.arange(256)
    
    
    for t in bins[1:-1]: 
        pc0 = np.sum(his[:t])
        pc1 = np.sum(his[t:])
        W0 = pc0 * mean_weight
        W1 = pc1 * mean_weight

        mu0 = np.sum(intensity_arr[:t]*his[:t]) / float(pc0)
        mu1 = np.sum(intensity_arr[t:]*his[t:]) / float(pc1)

        # Calculating between class variance
        sigma = W0 * W1 * (mu0 - mu1) ** 2

        if sigma > final_sigma:
            final_thresh = t
            final_sigma = sigma
        
    final_img = gray.copy()
    print("Threshold found by maximizing beetween class variance: ", final_thresh)
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    plt.figure()
    plt.axis("off")
    plt.imshow(final_img, cmap="gray")
    plt.title("Output using max_between_classv function")

# Calculating the threshold by minimizing the within class variance(sigma_w)
def min_within_classv(gray):
    
    total_pixels = gray.shape[0] * gray.shape[1]
    mean_weight = 1.0/total_pixels
    his, bins = np.histogram(gray, np.arange(0,257))
    final_thresh = -1
    final_sigma = np.inf # Assigning maximum value to final_sigma since it will be minimized
    intensity_arr = np.arange(256)
    
    for t in bins[1:-1]: 
        pc0 = np.sum(his[:t])
        pc1 = np.sum(his[t:])
        W0 = pc0 * mean_weight
        W1 = pc1 * mean_weight

        mu0 = np.sum(intensity_arr[:t]*his[:t]) / float(pc0)
        mu1 = np.sum(intensity_arr[t:]*his[t:]) / float(pc1)

        sigma_0_sq = np.sum((intensity_arr[:t]-mu0)**2 * his[:t]) / float(pc0)
        sigma_1_sq = np.sum((intensity_arr[t:]-mu1)**2 * his[t:]) / float(pc1)

        # Calculating within class variance
        sigma = W0 * sigma_0_sq + W1 * sigma_1_sq

        if sigma < final_sigma:
            final_thresh = t
            final_sigma = sigma
            
    final_img = gray.copy()
    print("\nThreshold found by minimizing within class variance: ", final_thresh)
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    plt.figure()
    plt.axis("off")
    plt.imshow(final_img, cmap="gray")
    plt.title("Output using min_within_classv function")
  
if __name__ == "__main__":
    
    start_time_max = time.time()
    max_between_classv(image)
    end_time_max = time.time()
    print("Time required using max_between_classv function: %.4f" % (end_time_max-start_time_max), "secs")
    
    start_time_min = time.time()
    min_within_classv(image)
    end_time_min = time.time()
    print("Time required using min_within_classv function: %.4f" % (end_time_min-start_time_min), "secs")
    
    otsu_library(image)