import cv2

face_cascade = cv2.CascadeClassifier("./files/haarcascade_frontalface_default.xml")

img = cv2.imread("./files/test6.jpg")

gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

for x, y, w, h in faces:
    img=cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

print(type(faces))
print(faces)


cv2.imshow("Face Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()