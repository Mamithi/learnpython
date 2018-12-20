import cv2, time

video= cv2.VideoCapture(0)
a = 1

while True:
    a=a+1
    check, frame = video.read() 

    print(check)
    print(frame)
    cv2.imshow("Capturing", frame)
    if(cv2.waitKey(1) == ord('q')):
        break

print(a)
video.release()
cv2.destroyAllWindows()