import cv2
import os

cap = cv2.VideoCapture(0)
cap.set(3,200) # 너비
cap.set(4,150) # 높이

face_detector = cv2.CascadeClassifier('./facetest/FaceDetection/Cascades/haarcascade_frontalface_default.xml')

# ID 등록
faceid = input('\n 생성ID를 입력하시오(숫자만 가능)>>>  ')
print("\n 캡쳐중입니다. 카메라를 보고 잠시 기다리시오...")

# 학습을 위해 캡처
count = 0
while(True):
    ret, img = cap.read()
    img = cv2.resize(img, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        
        # 화면을 캡처하여 그레이스케일로 폴더에 저장
        cv2.imwrite("dataset/User." + str(faceid) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    
    if cv2.waitKey(30) & 0xFF == 27:
        break
    # 30장의 사진을 찍으면 종료
    elif count >= 30: 
         break

print("\n 완료")
cap.release()
cv2.destroyAllWindows()
