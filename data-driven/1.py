import cv2
import numpy as np


image = cv2.imread('0000.png') 
final = np.array(image)
nonzero = np.nonzero(final)
for nz in nonzero:
    for n in nz:
    	print(n)
