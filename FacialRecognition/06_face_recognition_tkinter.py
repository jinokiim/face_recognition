from tkinter import *
import numpy as np
import cv2
import os
from PIL import Image


class App:
    def __init__(self):
        window = Tk()
        window.geometry('850x150')

        check_cap = Button(window, text='Check Cam', height=6, width=15, command=self.checkcam).pack(side=LEFT)
        
        detect_face = Button(window, text='Detect Face', height=6, width=15, command=self.detectface).pack(side=LEFT)
        
        sign_up = Button(window, text='Sign Up', height=6, width=15, command=self.signup).pack(side=LEFT)
        
        train_ing = Button(window, text='Training', height=6, width=15, command=self.training).pack(side=LEFT)
        
        face_recognition = Button(window, text='Recognition',height=6, width=15, command=self.recognition).pack(side=LEFT)
        
        window.mainloop()
        
    def checkcam(self):
        cap = cv2.VideoCapture(0)
        cap.set(3,200)
        cap.set(4,150)
        while(True):
            ret, img = cap.read()
            img = cv2.resize(img, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
            cv2.imshow('gray', gray)
    
            if cv2.waitKey(30) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        
    def detectface(self):
        faceCascade = cv2.CascadeClassifier('./facetest/FaceDetection/Cascades/haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        cap.set(3,200)
        cap.set(4,150)

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
        
    def signup(self):
        cap = cv2.VideoCapture(0)
        cap.set(3,200)
        cap.set(4,150)

        face_detector = cv2.CascadeClassifier('./facetest/FaceDetection/Cascades/haarcascade_frontalface_default.xml')

        faceid = input('\n 생성ID를 입력하시오(숫자만 가능)>>>  ')
        print("\n 캡쳐중입니다. 카메라를 보고 잠시 기다리시오...")

        count = 0
        while(True):
            ret, img = cap.read()
            img = cv2.resize(img, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                count += 1
        
                cv2.imwrite("dataset/User." + str(faceid) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('image', img)
    
            if cv2.waitKey(30) & 0xFF == 27:
                break
            elif count >= 30: 
                 break

        print("\n 완료")
        cap.release()
        cv2.destroyAllWindows()


        
        
    def training(self):
        path = 'dataset'
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("./facetest/FaceDetection/Cascades/haarcascade_frontalface_default.xml");


        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
            faceSamples=[]
            ids = []
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img,'uint8')
                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)
                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
            return faceSamples,ids
        print ("\n 학습중입니다... 조금만 기다시리시오...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        recognizer.write('trainer/trainer.yml')

        print("\n {0} 개의 얼굴을 학습했습니다. 프로그램을 종료합니다.".format(len(np.unique(ids))))
        
        
    def recognition(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "./facetest/FaceDetection/Cascades/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)
        font = cv2.FONT_HERSHEY_SIMPLEX

        id = 0
        names = ['None', 'Jinho', 'ChangJun', 'SH'] 

        # 비디오 실행
        cap = cv2.VideoCapture(0)
        cap.set(3,200) 
        cap.set(4,150) 
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
        
                if (confidence < 100):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                cv2.putText(img, str(id), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)  
    
            cv2.imshow('camera',img) 
            if cv2.waitKey(30) & 0xFF == 27:
                break

        print("\n 프로그램을 종료합니다.")
        cap.release()
        cv2.destroyAllWindows()


App()
