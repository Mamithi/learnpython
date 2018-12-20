import cv2
import glob
images = glob.glob("./images/*.jpg")

for image in images:
    img = cv2.imread(image, 0)
    re = cv2.resize(img, (500, 500))
    cv2.imshow("Hey", re)
    cv2.imwrite("resized"+image, re)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    