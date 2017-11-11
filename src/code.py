import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import sys
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
r=3.0 #number of squares
ref_ht=2.84 
rectangle_row=9
rectangle_col=6
square_size=6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
metre_pixel_x=0
metre_pixel_y=0

def squ_point(img, x, y, k):
	time_pass=25
	for i in range(time_pass):
		for j in range(time_pass):
			img[y-50+i, x-50+j] = np.array([10*k,50*k,0	])

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 	if event == cv2.EVENT_LBUTTONDOWN:
		pass
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))

def get_distance(image):
	global refPt
	refPt=[]

	while True:
		cv2.imshow("image", image)
		if(len(refPt)==2):
			break

		key = cv2.waitKey(1) & 0xFF

	if(len(refPt)==2):
		print refPt
		y_dist=abs(refPt[0][1]-refPt[1][1])
		x_dist=abs(refPt[0][0]-refPt[1][0])

		temp= (actual_dist/y_dist)*x_to_estimate
		return temp
	return 0

def chess_board_corners(gray):
	ret, corners = cv2.findChessboardCorners(gray, (rectangle_row,rectangle_col),None)
	corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
	global refPt
	refPt.append((corners2[0,0,0],corners2[0,0,1]))
	refPt.append((corners2[square_size-1,0,0],corners2[square_size-1,0,1]))
	refPt.append((corners2[rectangle_row*(square_size-1),0,0],corners2[rectangle_row*(square_size-1),0,1]))
	refPt.append((corners2[rectangle_row*(square_size-1)+square_size-1,0,0],corners2[rectangle_row*(square_size-1)+square_size-1,0,1]))
 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1000,800)
# cv2.namedWindow("image2",cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image2', 1000,800)
cv2.setMouseCallback("image", click_and_crop)
 
chess_board_corners(gray)

if len(refPt) == 4:
	# print refPt

	dist=(refPt[1][0]-refPt[0][0])#**2 + (refPt[0][1]-refPt[1][1])**2;
	# dist=sqrt(dist)
	print dist

	pt1=np.asarray(refPt,dtype=np.float32)
	print dist
	print pt1

	refPt[1]=(refPt[0][0]+dist,refPt[0][1])
	refPt[2]=(refPt[0][0],refPt[0][1]+dist)
	refPt[3]=(refPt[0][0]+dist,refPt[0][1]+dist)

	pt2=np.asarray(refPt,dtype=np.float32)
	print pt2


	M=cv2.getPerspectiveTransform(pt1,pt2)
	dst=cv2.warpPerspective(image,M,(image.shape[1],image.shape[0]))

	for i in range(4):
		squ_point(dst, int(pt2[i,0]), int(pt2[i,1]), i)
	cv2.imshow("image",dst)

	for i in range(4):
		squ_point(image, int(pt2[i,0]), int(pt2[i,1]), i)
	# cv2.imshow("image2",image)
	# print (get_height(dst))

	cv2.waitKey(0)
	cv2.imwrite('dst.jpg',dst)
	cv2.imwrite('image.jpg',image)
	


else:
	print "Didnt receive 4 points"
 
# close all open windows
cv2.destroyAllWindows()
