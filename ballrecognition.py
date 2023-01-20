import cv2
import numpy as np


# This function allows us to create a descending sorted list of contour areas.
def contour_area(contours):
     
    # create an empty list
    cnt_area = []
     
    # loop through all the contours
    for i in range(0,len(contours),1):
        # for each contour, use OpenCV to calculate the area of the contour
        cnt_area.append(cv2.contourArea(contours[i]))
 
    # Sort our list of contour areas in descending order
    list.sort(cnt_area, reverse=True)
    return cnt_area

def draw_bounding_box(contours, image, number_of_boxes=1):
    # Call our function to get the list of contour areas
    cnt_area = contour_area(contours)
    

    # Loop through each contour of ou  r image
    for i in range(0,len(contours),1):
        cnt = contours[i]
 
        # Only draw the the largest number of boxes
        if (cv2.contourArea(cnt) > cnt_area[number_of_boxes]):
             
            # Use OpenCV boundingRect function to get the details of the contour
            x,y,w,h = cv2.boundingRect(cnt)
        
            # Draw the bounding box
            image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    
 
    return image

def isolate_bounding_box(contours, image, number_of_boxes=1):
    # Call our function to get the list of contour areas
    cnt_area = contour_area(contours)
    
    
    # Loop through each contour of ou  r image
    for i in range(0,len(contours),1):
        cnt = contours[i]
 
        # Only draw the the largest number of boxes
        if (cv2.contourArea(cnt) > cnt_area[number_of_boxes]):
             
            # Use OpenCV boundingRect function to get the details of the contour
            x,y,w,h = cv2.boundingRect(cnt)
        
            # Draw the bounding box
            image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
            
            
            mask = np.zeros_like(image)
            mask = cv2.rectangle(mask, (x,y),(x+w,y+h), (255,255,255), -1)
            result = cv2.bitwise_and(image, mask)
    
    return result 


img = cv2.imread("testImage.jpg")
img = cv2.resize(img, (1280, 800))  

def get_pool_table_contours(image):
    lower_green = np.array([30,140,0])
    upper_green = np.array([100,255,255])

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower_green, upper_green)

    # Generate contours based on our mask
    contours,hierarchy = cv2.findContours(mask, 1, 2)
    return contours, hierarchy


def main():

    ksize = (10, 10)
    blur = cv2.blur(img, ksize)

    contours, hierachy = get_pool_table_contours(img)
    new_img = isolate_bounding_box(contours, img)

    while True:
        cv2.imshow("iamge", new_img)
        q = cv2.waitKey(1)
        if q==ord("q"):
            break

if __name__ == "__main__":
    main()
    