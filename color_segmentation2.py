import numpy as np
import cv2

img = cv2.imread("testImage.jpg")
Z = np.float32(img.reshape((-1,3)))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
_,labels,centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
labels = labels.reshape((img.shape[:-1]))
reduced = np.uint8(centers)[labels]

result = [np.hstack([img, reduced])]

mask = cv2.inRange(labels, 1, 1)
mask = np.dstack([mask]*3) # Make it 3 channel
ex_img = cv2.bitwise_and(img, mask)
ex_reduced = cv2.bitwise_and(reduced, mask)
result.append(np.hstack([ex_img, ex_reduced]))

cv2.imwrite('output.jpg', np.vstack(result))

cv2.imshow('img3',reduced)
cv2.imshow('img3',ex_reduced)



cv2.waitKey(0)

cv2.destroyAllWindows()