import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('./facetest/FaceDetection/Cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,200) # 너비
cap.set(4,150) # 높이

while True:
    ret, img = cap.read()
    img = cv2.resize(img, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

    for (x,y,w,h) in faces:
        # 사각형 생성
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
    cv2.imshow('video',img)

    if cv2.waitKey(30) & 0xFF == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
