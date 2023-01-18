import cv2
import numpy as np
url = "http://[2405:600:36f:659e:9d77:7da0:45d8:6ccd]:8081"
cp = cv2.VideoCapture(1)
while(True):
    camera, frame = cp.read()
    if frame is not None:
        cv2.imshow("Frame", frame)
    q = cv2.waitKey(1)
    if q==ord("q"):
        break
cv2.destroyAllWindows()