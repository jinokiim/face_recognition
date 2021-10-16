import cv2
import numpy as np

# 코 검출 데이터
nose_cascade = cv2.CascadeClassifier('./haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('./haarcascade_mcs_mouth.xml')

if nose_cascade.empty():
    raise IOError('Cannot find nose cascade classifier xml file!')
if mouth_cascade.empty():
    raise IOError('Cannot find mouth cascade classifier xml file!')

cap = cv2.VideoCapture(0)
ds_factor = 1.0

# 프로그램 실행
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 코 검출
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in nose_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        cv2.putText(frame, "please wear your mask", (20,50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255))
        break

    cv2.imshow('Nose Detector', frame)
    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
