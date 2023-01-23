import cv2
import numpy as np

img = cv2.imread("C:\\Users\\nickf\\OneDrive\\Bureau\\pool_table\\pool_table_image.jpg")

# It converts the BGR color space of image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
# Threshold of green in HSV space
lower_green = np.array([40, 40, 40])
upper_green = np.array([70, 255, 255])

# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_green, upper_green)
	
# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
result = cv2.bitwise_and(img, img, mask = mask)

cv2.imshow('frame', img)
cv2.imshow('mask', mask)
cv2.imshow('result', result)
	
cv2.waitKey(0)

cv2.destroyAllWindows()