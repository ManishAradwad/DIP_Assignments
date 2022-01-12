"""
Author: ManishAradwad
Date: 23/09/2021

Q3.

"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def shear(angle,x,y):
    
    tangent=math.tan(angle/2)
    new_x=round(x-y*tangent)
    new_y=y
    
    new_y=round(new_x*math.sin(angle)+new_y) 
    new_x=round(new_x-new_y*tangent) 
    
    return new_y,new_x

def rotate(image, angle):
    angle_deg = angle
    angle = math.radians(angle)                             
    cosine = math.cos(angle)
    sine = math.sin(angle)
    
    height = image.shape[0]                                  
    width = image.shape[1]                                   
    
    # Define the height and width of the new image that is to be formed
    new_height  = round(abs(height*cosine)+abs(width*sine))+1
    new_width  = round(abs(width*cosine)+abs(height*sine))+1
    
    # Define another image variable of dimensions of new_height and new _column filled with zeros
    output=np.zeros((new_height,new_width,image.shape[2]))    
    
    # Find the centre of the image about which we have to rotate the image
    original_centre_height = round(((height+1)/2)-1)    
    original_centre_width = round(((width+1)/2)-1)    
    
    # Find the centre of the new image that will be obtained
    new_centre_height = round(((new_height+1)/2)-1)        
    new_centre_width = round(((new_width+1)/2)-1)          
    
    
    for i in range(height):
        for j in range(width):
            
            # Co-ordinates of pixel with respect to the centre of original image
            y = height-1-i-original_centre_height                   
            x = width-1-j-original_centre_width 
    
            # Applying shear Transformation                     
            new_y, new_x = shear(angle,x,y)
    
            new_y = new_centre_height-new_y
            new_x = new_centre_width-new_x
            
            output[new_y,new_x,:]=image[i,j,:] 
    
    pil_img=Image.fromarray((output).astype(np.uint8)) 
    plt.figure()
    plt.imshow(pil_img)
    plt.title(f"Rotation by angle {-angle_deg}")

if __name__ == "__main__":    
    image = np.array(Image.open("./Bee.jpg"))
    angles= [90, 50]             
    plt.imshow(image)
    plt.title("Original Image")
    for angle in angles:
        rotate(image, angle)