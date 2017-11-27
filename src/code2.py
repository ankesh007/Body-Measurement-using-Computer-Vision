import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import segment
 
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
		# print refPt
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

	gray=np.copy(image)

	if(len(image.shape)>2):
		gray=cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY) 
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

	# print img.shape
	# if()
	img=img.astype(np.uint8)
	img_col = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	cv2.circle(img_col,(pt[0],pt[1]),10,(255,0,0),-1)
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
	# print y_head
	x_head=np.argmax(mask[y_head])
	return (x_head,y_head)

def first_sharp_fall(mask, x, y, win_size,thres):
	x_curr = x
	y0 = np.nonzero(mask[:,x_curr])[0][0]
	y0_diff = 10000
	x_curr = x+1*win_size
	y_curr = y0
	while True:
		if(len(np.nonzero(mask[:,x_curr])[0])==0):
			x_curr = x_curr-1*win_size
			break
		y_curr = np.nonzero(mask[:,x_curr])[0][0]
		y_diff = y_curr - y0
		if (y0_diff!=0):
			if((float(y_diff)/float(y0_diff))>thres):
				break
		x_curr=x_curr+1*win_size
		y0_diff=y_diff
		y0=y_curr

		if(x_curr==0):
			print("x reached 0")
			break
	return (x_curr,y_curr)

def analyze_chessboard(image,affine_correct_flag):
	clone = image.copy()
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	cv2.setMouseCallback(window_name1, click_and_crop)

	dst=np.copy(image) # created to ease affine_correct mode
	affine_correct_parameters=None
	if (affine_correct_flag=='True'):
		affine_correct_parameters=affine_correct_params(dst)

	gray2 = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY) 
	temp=chess_board_corners(dst,gray2,r2)

	ret, corners = cv2.findChessboardCorners(dst, (rectangle_row,rectangle_col),None)
	corners2 = cv2.cornerSubPix(gray2,corners,(11,11),(-1,-1),criteria)
	dst = cv2.drawChessboardCorners(dst, (9,6), corners2, ret)

	metre_pixel_x=(r2*ref_ht)/(abs(temp[0][0]-temp[1][0]))
	metre_pixel_y=(r2*ref_ht)/(abs(temp[0][1]-temp[2][1]))

	coordinate=[temp[0],temp[1]]
	# 6X6 square co-ord
	sep=((coordinate[1][0]-coordinate[0][0])/6.0)*9.0

	coordinate[0]=(max(0,int(coordinate[0][0]-2*sep)),0)
	coordinate[1]=(min(dst.shape[1],int(coordinate[1][0]+3.5*sep)),dst.shape[0])
	return metre_pixel_x,metre_pixel_y,coordinate,affine_correct_parameters

def getDistance(p1,p2):
	return (p1[0]-p2[0],p1[1]-p2[1])

def pixel_to_distance(p1,mx,my):
	return math.sqrt((p1[0]*mx)**2+(p1[1]*my)**2)


def measure_distance(segmented_image,metre_pixel_x,metre_pixel_y):
	head_pt = getHeadPoint(segmented_image)

	segmented_image = drawCircle(segmented_image, (head_pt[0],head_pt[1]), 1)
	cv2.imwrite('detected2.jpg', segmented_image)
	left_fall = first_sharp_fall(segmented_image, head_pt[0], head_pt[1], -1,12)
	right_fall = first_sharp_fall(segmented_image, head_pt[0], head_pt[1], 1,12)
	right_shoulder = first_sharp_fall(segmented_image, right_fall[0], right_fall[1], 20,1.5)
	left_shoulder = first_sharp_fall(segmented_image, left_fall[0], left_fall[1], -20,1.5)
	segmented_image = drawCircle(segmented_image, (right_fall[0], right_fall[1]), 1)
	segmented_image = drawCircle(segmented_image, (left_fall[0], left_fall[1]), 1)
	segmented_image = drawCircle(segmented_image, (right_shoulder[0], right_shoulder[1]), 1)
	segmented_image = drawCircle(segmented_image, (left_shoulder[0], left_shoulder[1]), 1)
	cv2.imwrite('detected.jpg', segmented_image)

	
	dist1=getDistance(left_shoulder,left_fall)
	dist1=pixel_to_distance(dist1,metre_pixel_x,metre_pixel_y)
	dist2=getDistance(right_shoulder,right_fall)
	dist2=pixel_to_distance(dist2,metre_pixel_x,metre_pixel_y)
	dist3=getDistance(left_fall,right_fall)
	dist3=pixel_to_distance(dist3,metre_pixel_x,metre_pixel_y)
	dist=dist1+dist2+dist3
	dist_tuple=dist1,dist2,dist3
	print dist,dist_tuple
	# dist=dist3+dist2+dist1
	# pixel_to_distance(dist,metre_pixel_x,metre_pixel_y)	


def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-i1", "--image1", required=True, help="Path to the checkboard_image")
	ap.add_argument("-i2", "--image2", required=True, help="Path to the arm_spread_image")
	ap.add_argument("-a", "--affine_mode", required=True, help="To perform Affine Corrections")
	args = vars(ap.parse_args())
	 
	# load the image, clone it, and setup the mouse callback function
	image = cv2.imread(args["image1"])
	arm_spread_image=cv2.imread(args["image2"])
	
	print "image",image.shape
	print "arm-spread",arm_spread_image.shape
	affine_correct_flag= (args["affine_mode"])
	# exit(1)

	metre_pixel_x,metre_pixel_y,coordinate,affine_correct_parameters=analyze_chessboard(image,affine_correct_flag)
	# print "Couldnt analyze"
	segmented_image=segment.segmenter(image)
	print "Segmentation Completed 1"

	segmented_arm_image=segment.segmenter(arm_spread_image)
	print "Segmentation Completed 2"

	cv2.imwrite("first.jpg",segmented_image)
	cv2.imwrite("second.jpg",segmented_arm_image)

	block_cut = np.zeros(segmented_image.shape)
	block_cut[coordinate[0][1]:coordinate[1][1],coordinate[0][0]:coordinate[1][0]] = 1
	segmented_image=segmented_image*block_cut

	if(affine_correct_flag=='True'):
		print affine_correct_flag
		segmented_image=affine_correct(segmented_image,affine_correct_parameters)
		print "Affine Corrected"

	# cv2.imwrite("affine_corrected.jpg",segmented_image)

	measure_distance(segmented_image,metre_pixel_x,metre_pixel_y)

if __name__=="__main__":
	main()
