import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import segment
# import sys
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
r1=5 #for affine correction
r2=2 #for measurement
ref_ht=2.84 
rectangle_row=9
rectangle_col=6
# square_size=int(r+1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
metre_pixel_x=0
metre_pixel_y=0
window_name1="image"

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
		cv2.imshow(window_name1, image)
		if(len(refPt)==2):
			break

		key = cv2.waitKey(1) & 0xFF
	cv2.destroyAllWindows()

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
def chess_board_corners(image,gray,r):
	square_size=int(r+1)
	ret, corners = cv2.findChessboardCorners(image, (rectangle_row,rectangle_col),None)
	corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
	coordinates=[]
	coordinates.append((corners2[0,0,0],corners2[0,0,1]))
	coordinates.append((corners2[square_size-1,0,0],corners2[square_size-1,0,1]))
	coordinates.append((corners2[rectangle_row*(square_size-1),0,0],corners2[rectangle_row*(square_size-1),0,1]))
	coordinates.append((corners2[rectangle_row*(square_size-1)+square_size-1,0,0],corners2[rectangle_row*(square_size-1)+square_size-1,0,1]))
	return coordinates
	# print coordinates

# receives an image and performs affine transform using chess_board_corners
def affine_correct_params(image):
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
	refPt=chess_board_corners(image,gray,r1)
	pt1=np.asarray(refPt,dtype=np.float32)
	dist=(refPt[1][0]-refPt[0][0])
	refPt[1]=(refPt[0][0]+dist,refPt[0][1])
	refPt[2]=(refPt[0][0],refPt[0][1]+dist)
	refPt[3]=(refPt[0][0]+dist,refPt[0][1]+dist)
	pt2=np.asarray(refPt,dtype=np.float32)
	M=cv2.getPerspectiveTransform(pt1,pt2)
	return M


def affine_correct(image,M=None):

	if M is None:
		M=affine_correct_params(image)

	image2=np.copy(image)

	if(len(image2)<3):
		image2=cv2.cvtColor(image2,cv2.COLOR_GRAY2RGB)
	
	dst=cv2.warpPerspective(image2,M,(image.shape[1],image.shape[0]))
	# dst=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
	return dst

# returns segmented image around refPt
def grub_cut(img,refPt):
	mask = np.zeros(img.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	rect = (refPt[0][0],refPt[0][1],refPt[1][0],refPt[1][1])
	cv2.imwrite("hey.jpg",img)
	cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img = img*mask2[:,:,np.newaxis]
	return img

def drawCircle(img, pt, state):
	print pt
	img_col = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	cv2.circle(img_col,(pt[0],pt[1]),20,(255,0,0),-1)
	if(state==0):
		while(1):
			cv2.imshow('img',img_col)
			k = cv2.waitKey(20) & 0xFF
			if k == 27:
				break
		cv2.destroyAllWindows()
		return img
	else:
		return cv2.cvtColor(img_col,cv2.COLOR_BGR2GRAY)

def getHeadPoint(mask):

	shape=mask.shape
	y_head=(np.nonzero(np.sum(mask,axis=1)))[0][0]
	print y_head
	x_head=np.argmax(mask[y_head])
	return (x_head,y_head)

def first_sharp_fall(mask, x, y, win_size,thres):

	y0 = np.nonzero(mask[:,x+1*win_size])[0][0]
	y0_diff = np.nonzero(mask[:,x+1*win_size])[0][0] - y
	x_curr = x+2*win_size
	while True:
		y_curr = np.nonzero(mask[:,x_curr])[0][0]
		y_diff = y_curr - y0
		print str(x_curr) + " " + str(y_diff)+" " + str(y0_diff)
		print (y_diff/y0_diff)

		if y0_diff!=0:
			if((float(y_diff)/float(y0_diff))>thres):
				break

		if(len(np.nonzero(mask[:,x_curr+1*win_size]))==0):
			break
		x_curr=x_curr+1*win_size
		y0_diff=y_diff
		y0=y_curr

		if(x_curr==0):
			print("x reached 0")
			break
	return (x_curr,y_curr)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-a", "--affine_mode", required=True, help="To perform Affine Corrections")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
print image.shape
# exit(1)
affine_correct_flag= (args["affine_mode"])

clone = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv2.namedWindow(window_name1,cv2.WINDOW_NORMAL)
# cv2.resizeWindow(window_name1, image.shape[0],image.shape[1])
cv2.setMouseCallback(window_name1, click_and_crop)

# print chess_board_corners(gray)
# exit(1)
dst=np.copy(image) # created to ease affine_correct mode
affine_correct_parameters=None
if (affine_correct_flag==True):
	affine_correct_parameters=affine_correct_params(dst)
gray2 = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY) 
temp=chess_board_corners(dst,gray2,r2)


ret, corners = cv2.findChessboardCorners(dst, (rectangle_row,rectangle_col),None)
corners2 = cv2.cornerSubPix(gray2,corners,(11,11),(-1,-1),criteria)
# print corners2
dst = cv2.drawChessboardCorners(dst, (9,6), corners2, ret)
# cv2.imshow(window_name1,dst)
metre_pixel_x=(r2*ref_ht)/(abs(temp[0][0]-temp[1][0]))
metre_pixel_y=(r2*ref_ht)/(abs(temp[0][1]-temp[2][1]))
#print get_distance(dst)
# sep=(coordinate[1][0]-coordinate[0][0])
coordinate=[temp[0],temp[1]]
# 6X6 square co-ord

sep=((coordinate[1][0]-coordinate[0][0])/6.0)*9.0
# print dst.shape[1]
# print sep
# exit(1)
coordinate[0]=(max(0,int(coordinate[0][0]-2*sep)),0)
coordinate[1]=(min(dst.shape[1],int(coordinate[1][0]+3.5*sep)),dst.shape[0])

block_cut = np.zeros(dst.shape)
block_cut[coordinate[0][1]:coordinate[1][1],coordinate[0][0]:coordinate[1][0],:] = 1
segmented_image=segment.segmenter(dst)
cv2.imwrite("sake.jpg",segmented_image)

if(affine_correct_flag=='True'):
	segmented_image=affine_correct(segmented_image,affine_correct_parameters)
	print "Affine Corrected"

head_pt = getHeadPoint(segmented_image)

left_fall = first_sharp_fall(segmented_image, head_pt[0], head_pt[1], -3,10)

right_fall = first_sharp_fall(segmented_image, head_pt[0], head_pt[1], 3,10)

right_shoulder = first_sharp_fall(segmented_image, right_fall[0], right_fall[1], 20,1.5)
left_shoulder = first_sharp_fall(segmented_image, left_fall[0], left_fall[1], -20,1.5)

segmented_image = drawCircle(segmented_image, (head_pt[0],head_pt[1]), 1)
segmented_image = drawCircle(segmented_image, (right_fall[0], right_fall[1]), 1)
segmented_image = drawCircle(segmented_image, (left_fall[0], left_fall[1]), 1)
segmented_image = drawCircle(segmented_image, (right_shoulder[0], right_shoulder[1]), 1)
segmented_image = drawCircle(segmented_image, (left_shoulder[0], left_shoulder[1]), 1)

cv2.imwrite('detected.jpg', segmented_image)
# dst2=grub_cut(np.copy(dst.astype(np.uint8)),coordinate)
# dst2=dst2*block_cut



# cv2.rectangle(dst2,(coordinate[0][0],coordinate[0][1]),(coordinate[1][0],coordinate[1][1]),(255,0,0),2)
# print block_cut.shape,"blockcut"
# print dst.shape,"dst"
# print block_cut.shape,"blockcut"
# cv2.waitKey(0)
# cv2.imwrite('dst.jpg',dst)
# cv2.imwrite('image.jpg',dst2)
# x_range = np.arange(coordinate[0][0], coordinate[1][0]+1)
# y_range = np.arange(coordinate[0][1], coordinate[1][1]+1)
# mesh = np.meshgrid(y_range, x_range)
# temp = np.zeros(dst.shape)
# temp[mesh] = 1
# dst = np.multiply(dst, temp)
 
# close all open windows
