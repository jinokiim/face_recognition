import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,200) # 너비
cap.set(4,150) # 높이
while(True):
    ret, img = cap.read()
    img = cv2.resize(img, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('img', img)
    cv2.imshow('gray', gray)
    
    if cv2.waitKey(30) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
