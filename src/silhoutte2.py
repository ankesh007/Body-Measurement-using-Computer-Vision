import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

img = cv2.imread(sys.argv[1])
print img.shape
# exit(1)
body_cascade=cv2.CascadeClassifier('haarcascade_fullbody.xml')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
body = body_cascade.detectMultiScale(gray,1.01,5)
lar = -1
p1_x = 100
p1_y = 100
p2_x = 1000
p2_y = 2000
# for (x,y,w,h) in body:
# 	if lar < w*h :
# 		lar = w*h
# 		p1_x = x
# 		p1_y = y
# 		p2_x = x+w
# 		p2_y = y+h

# print body
# print p1_x,

mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (p1_x,p1_y,p2_x,p2_y)
cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0),2)
# print rect
# exit(1)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()