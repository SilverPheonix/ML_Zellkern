#Packages to import

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

'''Can be deleted CV2 is not used in this code anymore'''
#image = cv2.imread("./output/test_masking/Figure_1.png", cv2.IMREAD_ANYCOLOR)
#cv2.imshow("Original", image)


im = Image.open('./output/test_masking/Figure_1.png').convert("RGBA")
pix = im.load()
#get size of image
width, height = (im.size)
#print (pix[309,214])  # Get the RGBA Value of the a pixel of an image

#convert image to array
image_array = np.array(im)

plt.imshow(image_array)
plt.show()

#create the empty mask
mask = np.zeros((height, width), dtype="uint8")

#iterate through the image to check pixl color value;
for y in range(height):
    for x in range(width):
        r, g, b, o = pix[x,y] #unpack the pixl
        if (r, g, b) == (0, 0, 0) or (r, g, b) == (255, 255, 255) or (r, g, b) == (95, 95, 95):  #Check if the pixel is white (border) or black (background)
            mask[y, x] = 0 
        else: mask[y,x]=255

#Plot Mask without Image underneath
plt.imshow(mask, cmap='gray')
plt.title("Mask without image")
plt.axis('off')
plt.show()


# Check if mask is boolean for masking
mask_bool = mask == 255

# Create a masked version
masked = np.zeros_like(image_array)
masked[mask_bool] = image_array[mask_bool]

# Show Mask on Image
plt.imshow(masked)
plt.title("Masked Image")
plt.axis('off')
plt.show()