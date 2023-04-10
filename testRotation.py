import cv2
import numpy as np

img = cv2.imread("1.jpg")
print(img.shape)
img_90 = cv2.flip(cv2.transpose(img), 1)
img_180 = cv2.flip(cv2.transpose(img_90), 1)
img_270 = cv2.flip(cv2.transpose(img_180), 1)

cv2.imshow('i', np.hstack([img, img_90, img_180, img_270]))
cv2.waitKey(0)
cv2.destroyAllWindows()
