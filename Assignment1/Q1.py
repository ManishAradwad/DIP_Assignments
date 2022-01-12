"""
Author: ManishAradwad
Date: 25/08/2021

Q1.

Histogram Computation: Compute the histogram of the image coins.png. Verify your result
using the MATLAB built-in function hist (or the corresponding function in python if you are using
python).
Function inputs: grayscale image, number of bins
Function outputs: bin centers, corresponding frequencies (from both your function
and MATLAB/Python function)

"""

# Importing libraries
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np

path = "./Coins.png"
image = skimage.io.imread(path)


def custom_plot(image, bins=16):
    
    # Getting all unique intensity values and their corresponding counts from binary image
    unique, counts = np.unique(image, return_counts=True)
    intensities = np.zeros(256)
    
    count_dict = dict(zip(unique, counts)) # Contains the count of intensities
    
    # Filling the intensities array with intensity counts. The index of intensities is intensity value.
    for key in count_dict:
        intensities[key] = int(count_dict[key])
    
    # Creating bins in the range [0, 255]
    bin_edges = np.linspace(0, 255, bins)
    
    # Calculating the total number of pixels having intensity in the range of corresponding bins
    bin_intensities = []
    for i in range(len(bin_edges)-1):
        temp = sum(intensities[int(bin_edges[i]):int(bin_edges[i+1])])
        bin_intensities.append(temp)
    
    # Since usage of any histogram function is not allowed, plotting a bar graph for comparison with matplotlib's hist() function.
    plt.bar((bin_edges[1:]+bin_edges[:-1])/2., bin_intensities, 10) # 10 is the thickness of each bar
    plt.xlabel("Intensity Value")
    plt.ylabel("Count")
    plt.title("Plot using custom_plot function")
    plt.show()
    
    print("Using Custom Function:")
    bincenters_freq(bin_edges, bin_intensities)


# Function to plot the histogram of given binary image
def inbuilt_plot(image, bins=16):
    
    # ravel() is needed since hist() expects a 1D array as input. Since img is a 2D array it is converted to 1D with ravel()
    n, binedges, patches = plt.hist(image.ravel(), bins=bins) 
    plt.xlabel("Intensity Value")
    plt.ylabel("Count")
    plt.title("Histogram using matplotlib's hist function")
    plt.show()
    
    print("\n\nUsing Library Functions:")
    bincenters_freq(binedges, n)
    print(binedges)
    
# Function to calculate and print the value of bin centres corresponding to each bin's intensity, given the bin edges
def bincenters_freq(binedges, freq):
    
    bincenters = np.mean(np.vstack([binedges[0:-1], binedges[1:]]), axis = 0)
    
    print("\nBin Center\t", "Frequency\n")
    for i in range(len(bincenters)):
        print("%5.2f \t\t  " %(bincenters[i]), freq[i])


if __name__ == "__main__":
    custom_plot(image, 32)
    inbuilt_plot(image, 32)