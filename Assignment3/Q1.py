# Importing libraries
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np

# Helper function for part a)
def img_generator(m, n, M, N, u_0, v_0):
    return np.sin((2*np.pi*u_0*m)/M + (2*np.pi*v_0*n)/N)

def part_a():
    image = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            image[m, n] = img_generator(m, n, M, N, u_0, v_0)
            
    dft_image = np.fft.fft2(image)
    shifted_image = np.fft.fftshift(dft_image)
    magnitude_spectrum = np.log(np.abs(shifted_image))
    
    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title("DFT Visualisation of sinusoidal image")
    
def part_b(image):
    image = np.array(image, dtype="uint32")
    P, Q = image.shape
    
    dft_image = np.fft.fft2(image)
    shifted_image = np.fft.fftshift(dft_image)
    D_0 = 100
    ILPF = np.zeros((P, Q))
    
    for u in range(P):
        for v in range(Q):
            D = np.sqrt((u-P/2)**2 + (v-Q/2)**2)
            if D <= D_0:
                ILPF[u, v] = 1
    
    ILPF = np.fft.fftshift(ILPF)
    final_img = np.fft.ifft2(np.multiply(shifted_image, ILPF))
    final_img = np.log(np.abs(final_img))     
    plt.figure()                    
    plt.imshow(final_img, cmap="gray")                       
    plt.title("Ideal Low Pass Filtered Image")

def part_c(image):
    image = np.array(image, dtype="uint32")
    P, Q = image.shape
    
    dft_image = np.fft.fft2(image)
    shifted_image = np.fft.fftshift(dft_image)
    D_0 = 100
    ILPF = np.zeros((P, Q))
    
    for u in range(P):
        for v in range(Q):
            D = np.sqrt((u-P/2)**2 + (v-Q/2)**2)
            ILPF[u, v] = np.exp(-D**2 / (2* D_0**2))
    
    ILPF = np.fft.fftshift(ILPF)
    final_img = np.fft.ifft2(np.multiply(shifted_image, ILPF))
    final_img = np.log(np.abs(final_img))    
    plt.figure()                    
    plt.imshow(final_img, cmap="gray")                       
    plt.title("Gaussian Low pass filtered image")

if __name__ == "__main__":
        
    M = 1001
    N = 1001
    u_0 = 100
    v_0 = 200

    part_a()
    image = skimage.io.imread("./characters.tif" )
    part_b(image)
    part_c(image)
    