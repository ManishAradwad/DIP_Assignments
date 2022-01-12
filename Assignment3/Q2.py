# Importing libraries
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import cv2

# Inverse Filter
def part_a(low_noise_fft, high_noise_fft, mat_fft):
    
    epsilon = 10
    mat_fft = 1 / (epsilon + mat_fft)
    unblur_low_noise_fft = low_noise_fft * mat_fft
    unblur_high_noise_fft = high_noise_fft * mat_fft
    
    unblur_low_noise_img = np.fft.ifft2(unblur_low_noise_fft).real
    unblur_high_noise_img = np.fft.ifft2(unblur_high_noise_fft).real
    
    plt.figure()
    plt.imshow(unblur_high_noise_img, cmap="gray")
    plt.title("Inverse filter output of High Noise Image")
    plt.figure()
    plt.imshow(unblur_low_noise_img, cmap="gray")
    plt.title("Inverse filter output of Low Noise Image")
    plt.show()

# Weiner Filter
def part_b(low_noise_fft, high_noise_fft, mat_fft):
        
    Sf = np.zeros((M, N))
    H = mat_fft
    
    for u in range(M):
        for v in range(N):
            if (u != 0 and v != 0):
                Sf[u, v] = 10**5 / np.sqrt(u**2 + v**2)
    
    Sw_low = 1
    Sw_high = 10
    
    D_low = (Sf * np.conj(H)) / (np.abs(H)**2 * Sf + Sw_low)  
    D_high = (Sf * np.conj(H)) / (np.abs(H)**2 * Sf + Sw_high)
    
    unblur_low_noise_fft = low_noise_fft * D_low
    unblur_high_noise_fft = high_noise_fft * D_high
    
    unblur_low_noise_img = np.abs(np.fft.ifft2(unblur_low_noise_fft))
    unblur_high_noise_img = np.abs(np.fft.ifft2(unblur_high_noise_fft))
    
    plt.figure()
    plt.imshow(unblur_high_noise_img, cmap="gray")
    plt.title("Weiner filter output of High Noise Image")
    plt.figure()
    plt.imshow(unblur_low_noise_img, cmap="gray")
    plt.title("Weiner filter output of Low Noise Image")

if __name__ == "__main__":
          
    mat = scipy.io.loadmat('./BlurKernel.mat')
    low_noise_img = skimage.io.imread("./Blurred_LowNoise.png")
    high_noise_img = skimage.io.imread("./Blurred_HighNoise.png")
    M, N = 688, 688
    
    mat = cv2.resize(mat["h"], (688, 688))
    mat_fft = np.fft.fft2(np.fft.ifftshift(mat))
    mat_fft[mat_fft < 0.1] = 0
    low_noise_fft = np.fft.fft2(low_noise_img)
    high_noise_fft = np.fft.fft2(high_noise_img)
    
    part_a(low_noise_fft, high_noise_fft, mat_fft)
    part_b(low_noise_fft, high_noise_fft, mat_fft)