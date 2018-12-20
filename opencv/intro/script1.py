import cv2

image = cv2.imread('./galaxy.jpg',0)

print(type(image))
print(image)
print(image.ndim)

resized_image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
cv2.imshow("Galaxy", resized_image)
cv2.imwrite("Galaxy_resized.jpg", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()