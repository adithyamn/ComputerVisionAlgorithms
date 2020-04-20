 #Using Hough Space to detect lines
 #Finds the best fitted pixel with the most intersected points of a different lines in an image space. 
 #Found via (theta,perpendicular dist) of different points which intersect at one point.

import cv2 
import numpy as np

def make_coordinates(image, line_parameters):
	slope, intercept =line_parameters
	y1 = image.shape[0]
	y2 = int(y1*(3/5))
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)

	
	return np.array([x1, y1, x2, y2])

def average_solpe_intercept(image,lines):
	left_fit = []
	right_fit = []
	for line in lines:
		x1,y1,x2,y2 = line.reshape(4)
		parameters = np.polyfit((x1,x2), (y1,y2), 1 )
		slope = parameters[0]
		intercept = parameters[1]
		if slope < 0:
			left_fit.append((slope, intercept))
		else : 
			right_fit.append((slope,intercept))
	left_fit_average = np.average(left_fit,axis = 0)
	right_fit_average = np.average(right_fit, axis = 0)
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)
	print(left_fit_average, 'left')
	print(right_fit_average, 'right')
	return np.array([left_line,right_line])


def canny(image):

	gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gs, (5,5), 0) 
	canny = cv2.Canny(blur, 50, 150)

	return canny

def display_lines(image,lines):	
	line_image = np.zeros_like(image)
	if lines is not None:
		for x1,y1,x2,y2 in lines:
			cv2.line(line_image, (x1,y1), (x2,y2),(255,9,9), 10 )
	return line_image  

def region_of_interest(image):

	height = image.shape[0]
	polygons = np.array([
		[(200,height),(1100,height),(550,250)]
		])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, polygons, 255)
	masked_image = cv2.bitwise_and(image, mask)

	return masked_image

#Main Code

image = cv2.imread('test_image.jpg')
image_copy = np.copy(image)
cann = canny(image_copy)
cropped_image = region_of_interest(cann )
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength =40, maxLineGap =5)
averaged_lines = average_solpe_intercept(image_copy, lines)
line_image = display_lines(image_copy, averaged_lines)
comb_image = cv2.addWeighted(image_copy, 0.8, line_image, 1, 1)
cv2.imshow("img",comb_image)

cv2.waitKey(0)   