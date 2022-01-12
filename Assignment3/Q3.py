# Importing libraries
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np

def part_a():
    
    noisy_book1 = skimage.io.imread("./noisy_book1.png")
    m, n = noisy_book1.shape
    
    nbhood = np.ones((3,3), dtype= int)
    nbhood = nbhood / 9
    
    result_img_mean = np.zeros((m, n))
    result_img_median = np.zeros((m, n))
    
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = noisy_book1[i-1:i+2, j-1:j+2]
            result_img_mean[i, j] = np.sum(np.multiply(temp, nbhood))
            result_img_median[i, j] = np.median(temp)
            
    plt.figure()
    plt.imshow(result_img_mean, cmap="gray")
    plt.title("Q2.a) Mean Filtered Image")
    plt.figure()
    plt.imshow(result_img_median, cmap="gray")
    plt.title("Q2.a) Median Filtered Image")
    plt.show()
    
def part_b():

    noisy_book2 = skimage.io.imread("./noisy_book2.png")
    
    # Bilateral Filter
    bf_filtered_img = filter_bilateral(noisy_book2, 5)
    plt.figure()
    plt.imshow(bf_filtered_img, cmap="gray")
    plt.title("Q2.b) Bilateral Filtered Image")
    
    # Gaussian Filter
    gaussian_filtered_img = filter_gaussian(noisy_book2, 5)
    plt.figure()
    plt.imshow(gaussian_filtered_img, cmap="gray")
    plt.title("Q2.b) Gaussian Filtered Image")
    

# Bilateral Filter
def filter_bilateral( img_in, sigma_s, reg_constant=1e-8 ):

    gaussian = lambda r2, sigma: (np.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0
    win_width = int( 3*sigma_s+1 )
    wgt_sum = np.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            w = gaussian( shft_x**2+shft_y**2, sigma_s )
            off = np.roll(img_in, [shft_y, shft_x], axis=[0,1])
            tw = w*gaussian((off-img_in)**2, sigma_s)
            result += off*tw
            wgt_sum += tw

    return result/wgt_sum

# Gaussian Filter
def filter_gaussian(image, sigma):
    
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2
    
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2
    
    im_filtered = np.zeros_like(image, dtype=np.float32)
    im_filtered[:, :] = convolution(image[:, :], gaussian_filter)
    return (im_filtered.astype(np.uint8))

# Helper function for gaussian filter
def convolution(oldimage, kernel):
    
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    
    image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2),(kernel_w // 2, kernel_w // 2)), mode='constant', constant_values=0).astype(np.float32)
        
    h = kernel_h // 2
    w = kernel_w // 2
    
    image_conv = np.zeros(image_pad.shape)
    
    for i in range(h, image_pad.shape[0]-h):
        for j in range(w, image_pad.shape[1]-w):
            
            x = image_pad[i-h:i-h+kernel_h, j-w:j-w+kernel_w]
            x = x.flatten()*kernel.flatten()
            image_conv[i][j] = x.sum()
    
    h_end = -h
    w_end = -w
    
    if(h == 0):
        return image_conv[h:,w:w_end]
    if(w == 0):
        return image_conv[h:h_end,w:]
    
    return image_conv[h:h_end,w:w_end]

if __name__ == "__main__":
    
    noisy_book1 = skimage.io.imread("./noisy_book1.png")
    noisy_book2 = skimage.io.imread("./noisy_book2.png")
    
    part_a()
    part_b()
    