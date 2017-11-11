import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
# import sys
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
r=3.0 #number of squares
ref_ht=2.84 
rectangle_row=9
rectangle_col=6
square_size=int(r+1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
metre_pixel_x=0
metre_pixel_y=0

def squ_point(img, x, y, k):
	time_pass=50
	for i in range(time_pass):
		for j in range(time_pass):
			img[y-25+i, x-25+j] = np.array([10*k,50*k,0	])

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 	if event == cv2.EVENT_LBUTTONDOWN:
		pass
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))

# returns real-world distance between 2 points selected in image
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
		pixel_dist_y=abs(refPt[0][1]-refPt[1][1])
		pixel_dist_x=abs(refPt[0][0]-refPt[1][0])

		actual_y=metre_pixel_y*pixel_dist_y
		actual_x=metre_pixel_x*pixel_dist_x

		actual_dist=math.sqrt(actual_y**2 + actual_x**2)
		return actual_dist

	return 0

# returns 4 points at square_size of checkboard 
def chess_board_corners(gray):
	ret, corners = cv2.findChessboardCorners(gray, (rectangle_row,rectangle_col),None)
	corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
	coordinates=[]
	coordinates.append((corners2[0,0,0],corners2[0,0,1]))
	coordinates.append((corners2[square_size-1,0,0],corners2[square_size-1,0,1]))
	coordinates.append((corners2[rectangle_row*(square_size-1),0,0],corners2[rectangle_row*(square_size-1),0,1]))
	coordinates.append((corners2[rectangle_row*(square_size-1)+square_size-1,0,0],corners2[rectangle_row*(square_size-1)+square_size-1,0,1]))
	return coordinates

# receives an image and performs affine transform using chess_board_corners
def affine_correct(image):

	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
	refPt=chess_board_corners(gray)
	pt1=np.asarray(refPt,dtype=np.float32)
	dist=(refPt[1][0]-refPt[0][0])
	refPt[1]=(refPt[0][0]+dist,refPt[0][1])
	refPt[2]=(refPt[0][0],refPt[0][1]+dist)
	refPt[3]=(refPt[0][0]+dist,refPt[0][1]+dist)
	pt2=np.asarray(refPt,dtype=np.float32)
	M=cv2.getPerspectiveTransform(pt1,pt2)
	dst=cv2.warpPerspective(image,M,(image.shape[1],image.shape[0]))
	return dst

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1000,800)
cv2.setMouseCallback("image", click_and_crop)

# print chess_board_corners(gray)
dst=affine_correct(image)
gray2 = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY) 
temp=chess_board_corners(gray2)
# print temp
# for i in range(4):
# 	squ_point(dst, int(temp[i][0]), int(temp[i][1]), i)
cv2.imshow("image",dst)
metre_pixel_x=(r*ref_ht)/(abs(temp[0][0]-temp[1][0]))
metre_pixel_y=(r*ref_ht)/(abs(temp[0][1]-temp[2][1]))
print get_distance(dst)
# print metre_pixel_y
# print metre_pixel_x

cv2.waitKey(0)
cv2.imwrite('dst.jpg',dst)
cv2.imwrite('image.jpg',image)
 
# close all open windows
cv2.destroyAllWindows()
