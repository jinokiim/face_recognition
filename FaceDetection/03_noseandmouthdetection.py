import cv2
import numpy as np

# 얼굴, 코, 입 검출을 위한 케스케이드 분류기 생성
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('./haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('./haarcascade_mcs_mouth.xml')

if nose_cascade.empty():
    raise IOError('Cannot find nose cascade classifier xml file!')
if mouth_cascade.empty():
    raise IOError('Cannot find mouth cascade classifier xml file!')

cap = cv2.VideoCapture(0)

# 프로그램 실행
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 얼굴 검출
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(80,80))
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
            roi = gray[y:y+h, x:x+w]
            
            # 코 검출
            nose = nose_cascade.detectMultiScale(roi)
            for i, (nx,ny,nw,nh) in enumerate(nose):
                if i>=2:
                    break
                cv2.rectangle(img[y:y+h, x:x+w],(nx,ny), (nx+nw,ny+nh), (0,0,255),2)
                cv2.putText(img, "please wear your mask", (20,50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255))
            # 입 검출
            mouth = mouth_cascade.detectMultiScale(roi)
            for i, (mx,my,mw,mh) in enumerate(nose):
                if i>=2:
                    break
                cv2.rectangle(img[y:y+h, x:x+w],(mx,my), (mx+mw,my+mh), (0,0,255),2)
                cv2.putText(img, "please wear your mask", (20,50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255))
        cv2.imshow('Detector', img)
    else:
        break
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
