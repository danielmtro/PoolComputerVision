import numpy as np
import cv2

# Read input
color = cv2.imread("C:\\Users\\nickf\\OneDrive\\Bureau\\pool_table\\pool_table_image.jpg")

color = cv2.resize(color, (0, 0), fx=0.15, fy=0.15)

color_blur = cv2.GaussianBlur(color,(3,3), 0)
# RGB to gray
gray = cv2.cvtColor(color_blur, cv2.COLOR_BGR2GRAY)
cv2.imwrite('output/gray.png', gray)
# cv2.imwrite('output/thresh.png', thresh)
# Edge detection
edges = cv2.Canny(gray, 700, 720, apertureSize=5, L2gradient = True)
# Save the edge detected image
cv2.imwrite('output/edges.png', edges)


#cv2.imshow('image1',gray)


#cv2.imshow('image2',color)
#cv2.imshow('image3',color_blur)


cv2.imshow('image',edges)

cv2.waitKey(0)
cv2.destroyAllWindows()