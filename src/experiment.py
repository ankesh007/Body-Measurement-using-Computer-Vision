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
		refPt.append4((x, y))

# returns real-world distance between 2 points selected in image
def get_distance(image):
	global refPt
	refPt=[]

	while True:
		cv2.imshow(window_name1, image)
		if(len(refPt)==2):
			break
		# print refPt
		k = cv2.waitKey(1) & 0xFF
	cv2.destroyAllWindows()

	if(len(refPt)==2):
		# print refPt
		pixel_dist_y=abs(refPt[0][1]-refPt[1][1])
		pixel_dist_x=abs(refPt[0][0]-refPt[1][0])

		actual_y=metre_pixel_y*pixel_dist_y
		actual_x=metre_pixel_x*pixel_dist_x

		actual_dist=math.sqrt(actual_y**2 + actual_x**2)
		print actual_dist
		return actual_dist

	return 0

def get_points(img):
    points= []
    img_to_show = img.copy()
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img_to_show,(x,y),2,(255,0,0),-1)
            points.append([x,y])
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', img.shape[0],img.shape[1])
    cv2.setMouseCallback('image',draw_circle)
    while(1):
        cv2.imshow('image',img_to_show)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    return points

def get_real_world_distance(points,m_x,m_y):
	pixel_dist_y=abs(points[0][1]-points[1][1])
	pixel_dist_x=abs(points[0][0]-points[1][0])
	actual_y=m_y*pixel_dist_y
	actual_x=m_x*pixel_dist_x
	actual_dist=math.sqrt(actual_y**2 + actual_x**2)
	

def get_waist(img,m_x,m_y):
	points=get_points(img)
	actual_dist=get_real_world_distance(points,m_x,m_y)
	print actual_dist
	return actual_dist


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
	rect = (refPt[0][0]+5,refPt[0][1]+5,refPt[1][0]-5,refPt[1][1]-5)
	print rect
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
	cv2.circle(img_col,(pt[0],pt[1]),18,(255,0,255),-1)
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

def get_wrist(mask):
	thres = 20 * 255
	wrist_x_left = np.nonzero(np.sum(mask,axis=0) > thres)[0][0] 
	wrist_y_left = np.argmax(mask[:,wrist_x_left])
	circled = drawCircle(mask,(wrist_x_left,wrist_y_left),1)
	nonzero = len(np.nonzero(np.sum(mask,axis=0) > thres)[0])
	wrist_x_right = np.nonzero(np.sum(mask,axis=0) > thres)[0][nonzero - 1] 
	wrist_y_right = np.argmax(mask[:,wrist_x_right])
	circled = drawCircle(circled,(wrist_x_right,wrist_y_right),1)
	cv2.imwrite("detectedwrist.jpg",circled)
	return (wrist_x_left,wrist_y_left),(wrist_x_right,wrist_y_right)


	# print "hi"


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


def measure_distance(segmented_image,segmented_arm_image,arm_spread_image,waist_image,image,metre_pixel_x,metre_pixel_y):
	waist_a = get_points(arm_spread_image)
	waist_b = get_points(waist_image)
	dist1=getDistance(waist_a[0],waist_a[1])
	dist1=pixel_to_distance(dist1,metre_pixel_x,metre_pixel_y)
	dist2=getDistance(waist_b[0],waist_b[1])
	dist2=pixel_to_distance(dist2,metre_pixel_x,metre_pixel_y)
	dist1 = dist1/2
	dist2 = dist2/2
	perimeter = 2 * 3.1415 * math.sqrt((dist1*dist1 + dist2*dist2)/2)
	print "waist",perimeter

	head_pt = getHeadPoint(segmented_image)

	segmented_image = drawCircle(segmented_image, (head_pt[0],head_pt[1]), 1)
	cv2.imwrite('detected2.jpg', segmented_image)

	left_fall = first_sharp_fall(segmented_image, head_pt[0], head_pt[1], -2,14)
	right_fall = first_sharp_fall(segmented_image, head_pt[0], head_pt[1], 2,14)
	segmented_image = drawCircle(segmented_image, (right_fall[0], right_fall[1]), 1)
	segmented_image = drawCircle(segmented_image, (left_fall[0], left_fall[1]), 1)
	points = get_points(segmented_image)
	if len(points) != 0:
		left_fall = points[0]
		right_fall = points[1]
	
	right_shoulder = first_sharp_fall(segmented_image, right_fall[0], right_fall[1], 20,1.5)
	left_shoulder = first_sharp_fall(segmented_image, left_fall[0], left_fall[1], -20,1.5)
	segmented_image = drawCircle(segmented_image, (right_shoulder[0], right_shoulder[1]), 1)
	segmented_image = drawCircle(segmented_image, (left_shoulder[0], left_shoulder[1]), 1)
	points = get_points(segmented_image)
	if len(points) != 0:
		left_shoulder = points[0]
		right_shoulder = points[1]

	left_wrist,right_wrist = get_wrist(segmented_arm_image)
	cv2.imwrite('detected_r.jpg', np.concatenate((image,cv2.cvtColor(segmented_image,cv2.COLOR_GRAY2RGB)),axis=1))
	segmented_image = drawCircle(segmented_arm_image, (left_wrist[0], left_wrist[1]), 1)
	segmented_image = drawCircle(segmented_arm_image, (right_wrist[0], right_wrist[1]), 1)
	points = get_points(segmented_image)
	if len(points) != 0:
		left_wrist = points[0]
		right_wrist = points[1]

	cv2.imwrite('detected_wrist.jpg', np.concatenate((arm_spread_image,cv2.cvtColor(segmented_arm_image,cv2.COLOR_GRAY2RGB)),axis=1))
	print segmented_image.shape
	print image.shape


	
	dist1=getDistance(left_shoulder,left_fall)
	dist1=pixel_to_distance(dist1,metre_pixel_x,metre_pixel_y)
	dist2=getDistance(right_shoulder,right_fall)
	dist2=pixel_to_distance(dist2,metre_pixel_x,metre_pixel_y)
	dist3=getDistance(left_fall,right_fall)
	dist3=pixel_to_distance(dist3,metre_pixel_x,metre_pixel_y)
	dist4=getDistance(left_wrist,left_shoulder)
	dist4=pixel_to_distance(dist4,metre_pixel_x,metre_pixel_y)
	dist5=getDistance(right_wrist,right_shoulder)
	dist5=pixel_to_distance(dist5,metre_pixel_x,metre_pixel_y)
	dist_sleeve = (dist5+dist4)/2.0
	dist=dist1+dist2+dist3
	dist_tuple=dist1,dist2,dist3
	print "Shoulder Length",dist
	print "Sleeve Length", dist_sleeve,(dist4,dist5)
	# dist=dist3+dist2+dist1
	# pixel_to_distance(dist,metre_pixel_x,metre_pixel_y)	

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-i1", "--image1", required=True, help="Path to the checkboard_image")
	ap.add_argument("-i2", "--image2", required=True, help="Path to the arm_spread_image")
	ap.add_argument("-i3", "--image3", required=True, help="Path to the waist_image")
	ap.add_argument("-a", "--affine_mode", required=True, help="To perform Affine Corrections")
	args = vars(ap.parse_args())
	 
	# load the image, clone it, and setup the mouse callback function
	image = cv2.imread(args["image1"])
	image22 = np.copy(image)
	arm_spread_image=cv2.imread(args["image2"])
	waist_image = cv2.imread(args["image3"])
	
	affine_correct_flag= (args["affine_mode"])

	metre_pixel_x,metre_pixel_y,coordinate,affine_correct_parameters=analyze_chessboard(image,affine_correct_flag)
	
	segmented_image=segment.segmenter(image)
	print "Segmentation Completed 1"

	segmented_arm_image=segment.segmenter(arm_spread_image)
	print "Segmentation Completed 2"

	# cv2.imwrite("first.jpg",np.concatenate((image,cv2.cvtColor(segmented_image,cv2.COLOR_GRAY2RGB)),axis=1))
	# cv2.imwrite("second.jpg",np.concatenate((arm_spread_image,cv2.cvtColor(segmented_arm_image,cv2.COLOR_GRAY2RGB)),axis=1))
	
	image2=affine_correct(image,affine_correct_parameters)
	# cut_image = grub_cut(image2,[(0,0),(image.shape[1],image.shape[0])])
	image2=cv2.rectangle(image2,(coordinate[0][0],coordinate[0][1]),(coordinate[1][0],coordinate[1][1]),(255,0,0),3)
	# cv2.imwrite('cascade1.jpg',np.concatenate((image,image2),axis=1))
	# exit(0)

	block_cut = np.zeros(segmented_image.shape)
	block_cut[coordinate[0][1]:coordinate[1][1],coordinate[0][0]:coordinate[1][0]] = 1
	segmented_image=segmented_image*block_cut

	if(affine_correct_flag=='True'):
		image2=affine_correct(image,affine_correct_parameters)
		cv2.imwrite('affine_correction_3.jpg',np.concatenate((image,image2),axis=1))
		# exit(0)
		arm_spread_image=affine_correct(arm_spread_image,affine_correct_parameters)
		waist_image=affine_correct(waist_image,affine_correct_parameters)
		segmented_image=affine_correct(segmented_image,affine_correct_parameters)
		print "Affine Corrected"

	# detect_wrist(segmented_arm_image)
	
	# cv2.imwrite("affine_corrected.jpg",segmented_image)

	measure_distance(segmented_image,segmented_arm_image,arm_spread_image,waist_image,image22,metre_pixel_x,metre_pixel_y)

if __name__=="__main__":
	main()
