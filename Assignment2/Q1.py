"""
Author: ManishAradwad
Date: 23/09/2021

Q1.

"""

# Importing libraries
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import numpy as np
    
# This performs all contrast enhancement algorithms 
def contrast_enhancement(image):
 
    image = skimage.io.imread("./" + image)
    image = np.array(image, dtype="uint32")
    og_img = image
    image = image/255
    
    # Full Scale Contrast Stretch
    a_max = np.amax(image)
    a_min = np.amin(image)
    fscs_img = (image-a_min)/(a_max-a_min)
    plots(image, fscs_img, "Full Scale Contrast Stretch")

    # Non-linear Contrast Stretch
    nlcs_img = np.exp(image) / 255
    nlcs_img[nlcs_img < 0] = 0
    nlcs_img[nlcs_img > 1] = 1
    plots(image, nlcs_img, "Non-linear Contrast Stretch")
    
    # Histogram Equalization followed by Full Scale Contrast Stretch
    count = np.histogram(image.ravel(), bins=256)
    count = count[0]
    pdf = count // sum(count)
    cdf = np.cumsum(pdf)
    new_img = image.copy()

    for i in range(256):
        new_img[new_img == i] = cdf[i]
    
    a_max = np.amax(new_img)
    a_min = np.amin(new_img)
    he_img = (new_img-a_min)/(a_max-a_min)
    plots(image, he_img, "Histogram Equalization with Full Scale Contrast Stretch")
    
    # CLAHE - No Overlap
    clahe_img = clahe(og_img, 4, 128, 0, 0)
    plots(image, clahe_img, "CLAHE - No Overlap")  
    
# Function for plotting
def plots(img, new_img, title):    
    
    fig = plt.figure()
    fig.suptitle(title)
    plt.subplot(2,2,1)
    plt.axis("off")
    plt.imshow(img, vmin=0, vmax=1, cmap="gray")
    plt.subplot(2,2,2)
    plt.hist(img.ravel(), bins=32)
    plt.subplot(2,2,3)
    plt.axis("off")
    plt.imshow(new_img, cmap="gray")
    plt.subplot(2,2,4)
    plt.hist(new_img.ravel(), bins=32)
    plt.show()

# Function used for interpolation after clahe to reduce boundary artifacts
def interpolate(subBin,LU,RU,LB,RB,subX,subY):
    
    subImage = np.zeros(subBin.shape)
    num = subX*subY
    for i in range(subX):
        inverseI = subX-i
        for j in range(subY):
            inverseJ = subY-j
            val = subBin[i,j].astype(int)
            subImage[i,j] = np.floor((inverseI*(inverseJ*LU[val] + j*RU[val])+ i*(inverseJ*LB[val] + j*RB[val]))/float(num))
    return subImage

# Function to perform CLAHE without any overlap in blocks
def clahe(img, clipLimit, nrBins=128, nrX=0, nrY=0):
    
    h,w = img.shape
    
    nrBins = max(nrBins,128)
    if nrX==0:
        xsz = 8
        ysz = 8
        nrX = np.ceil(h/xsz).astype(int)
        excX= int(xsz*(nrX-h/xsz))
        nrY = np.ceil(w/ysz).astype(int)
        excY= int(ysz*(nrY-w/ysz))
        if excX!=0:
            img = np.append(img,np.zeros((excX,img.shape[1])).astype(int),axis=0)
        if excY!=0:
            img = np.append(img,np.zeros((img.shape[0],excY)).astype(int),axis=1)
    else:
        xsz = round(h/nrX)
        ysz = round(w/nrY)
    
    nrPixels = xsz*ysz
    claheimg = np.zeros(img.shape)
    
    minVal = 0 
    maxVal = 255 
    
    binSz = np.floor(1+(maxVal-minVal)/float(nrBins))
    LUT = np.floor((np.arange(minVal,maxVal+1)-minVal)/float(binSz))
    
    bins = LUT[img]
    hist = np.zeros((nrX,nrY,nrBins))
    for i in range(nrX):
        for j in range(nrY):
            bin_ = bins[i*xsz:(i+1)*xsz,j*ysz:(j+1)*ysz].astype(int)
            for i1 in range(xsz):
                for j1 in range(ysz):
                    hist[i,j,bin_[i1,j1]]+=1
    
    if clipLimit>0:
        for i in range(nrX):
            for j in range(nrY):
                nrExcess = 0
                for nr in range(nrBins):
                    excess = hist[i,j,nr] - clipLimit
                    if excess>0:
                        nrExcess += excess
                
                binIncr = nrExcess/nrBins
                upper = clipLimit - binIncr
                for nr in range(nrBins):
                    if hist[i,j,nr] > clipLimit:
                        hist[i,j,nr] = clipLimit
                    else:
                        if hist[i,j,nr]>upper:
                            nrExcess += upper - hist[i,j,nr]
                            hist[i,j,nr] = clipLimit
                        else:
                            nrExcess -= binIncr
                            hist[i,j,nr] += binIncr
                
                if nrExcess > 0:
                    stepSz = max(1,np.floor(1+nrExcess/nrBins))
                    for nr in range(nrBins):
                        nrExcess -= stepSz
                        hist[i,j,nr] += stepSz
                        if nrExcess < 1:
                            break
    
    map_ = np.zeros((nrX,nrY,nrBins))
    scale = (maxVal - minVal)/float(nrPixels)
    for i in range(nrX):
        for j in range(nrY):
            sum_ = 0
            for nr in range(nrBins):
                sum_ += hist[i,j,nr]
                map_[i,j,nr] = np.floor(min(minVal+sum_*scale,maxVal))
    
    xI = 0
    for i in range(nrX+1):
        if i==0:
            subX = int(xsz/2)
            xU = 0
            xB = 0
        elif i==nrX:
            subX = int(xsz/2)
            xU = nrX-1
            xB = nrX-1
        else:
            subX = xsz
            xU = i-1
            xB = i
        
        yI = 0
        for j in range(nrY+1):
            if j==0:
                subY = int(ysz/2)
                yL = 0
                yR = 0
            elif j==nrY:
                subY = int(ysz/2)
                yL = nrY-1
                yR = nrY-1
            else:
                subY = ysz
                yL = j-1
                yR = j
            UL = map_[xU,yL,:]
            UR = map_[xU,yR,:]
            BL = map_[xB,yL,:]
            BR = map_[xB,yR,:]
            subBin = bins[xI:xI+subX,yI:yI+subY]
            subImage = interpolate(subBin,UL,UR,BL,BR,subX,subY)
            claheimg[xI:xI+subX,yI:yI+subY] = subImage
            yI += subY
        xI += subX
    
    if excX==0 and excY!=0:
        return claheimg[:,:-excY]
    elif excX!=0 and excY==0:
        return claheimg[:-excX,:]
    elif excX!=0 and excY!=0:
        return claheimg[:-excX,:-excY]
    else:
        return claheimg
      
if __name__ == "__main__":
    
    inputs = [
                "IIScMainBuilding_LowContrast.png", 
                "LowLight_2.png", 
                "LowLight_3.png", 
                "Hazy.png", 
                "StoneFace.png"
              ]
              
    for ip in inputs:
        contrast_enhancement(ip)