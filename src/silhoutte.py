import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

threshold=70
window_name1="image"

image=cv2.imread(sys.argv[1])
background=cv2.imread(sys.argv[2])

diff=(image-background)

flag=diff>threshold 
flag=(flag[:,:,0]*flag[:,:,1]*flag[:,:,2])>0
flag=flag[:,:,np.newaxis]
flag=np.concatenate((flag,flag,flag),axis=2)
print flag.shape
image=image*flag

cv2.namedWindow(window_name1,cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name1, image.shape[0],image.shape[1])
# cv2.imshow(window_name1, image)
cv2.imwrite('silhoutte.jpg',image)
