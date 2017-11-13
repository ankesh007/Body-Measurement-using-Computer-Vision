import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

img = cv2.imread(sys.argv[1])
print img.shape
window_name1="image"
# exit(1)
refPt = []
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 	if event == cv2.EVENT_LBUTTONDOWN:
		pass
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))
		print x,y

# body_cascade=cv2.CascadeClassifier('haarcascade_fullbody.xml')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# body = body_cascade.detectMultiScale(gray,1.01,5)

# lar = -1
# p1_x = 100
# p1_y = 100
# p2_x = int((img.shape[0]))
# p2_y = int(img.shape[1]*0.75)
cv2.namedWindow(window_name1,cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name1, img.shape[0],img.shape[1])
cv2.setMouseCallback(window_name1, click_and_crop)
while True:
	cv2.imshow(window_name1, img)
	if(len(refPt)==2):
		break
	key = cv2.waitKey(1) & 0xFF


print "Yo"
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (refPt[0][0],refPt[0][1],refPt[1][0],refPt[1][1])


print rect
# print rect
# exit(1)
cv2.destroyAllWindows()
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0),2)

x_range = np.arange(refPt[0][0], refPt[1][0]+1)
y_range = np.arange(refPt[0][1], refPt[1][1]+1)
mesh = np.meshgrid(y_range, x_range)
temp = np.zeros(img.shape)
temp[mesh] = 1
img = np.multiply(img, temp)

cv2.imwrite('image2.jpg',img)
# plt.imshow(img),plt.colorbar(),plt.show()