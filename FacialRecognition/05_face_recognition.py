import cv2
import numpy as np
import os 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "./facetest/FaceDetection/Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
# id에 맞게 리스트 구현(id가 1인사람의 이름을 list[1] 자리에 입력)
names = ['None', 'Jinho', 'ChangJun', 'SH'] 

# 비디오 실행
cap = cv2.VideoCapture(0)
cap.set(3,200) # 너비
cap.set(4,150) # 높이
# 얼굴 인식할수있는 최소크기 설정
minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)

while True:
    ret, img =cap.read()
    img = cv2.resize(img, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5,minSize = (int(minW), int(minH)))
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # id출력
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        # 관련 이름 입력
        cv2.putText(img, str(id), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # 정확도 입력
        cv2.putText(img, str(confidence), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 
    if cv2.waitKey(30) & 0xFF == 27:
        break

print("\n 프로그램을 종료합니다.")
cap.release()
cv2.destroyAllWindows()
