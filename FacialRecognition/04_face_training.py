import cv2
import numpy as np
from PIL import Image
import os

# 저장된 경로 설정
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("./facetest/FaceDetection/Cascades/haarcascade_frontalface_default.xml");

# 이미지 학습 관련 함수
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

# model을 trainer폴더에 trainer.yml로 저장
recognizer.write('trainer/trainer.yml')

# 몇개 종류의 얼굴을 학습했는지 출력
print("\n {0} 개의 얼굴을 학습했습니다. 프로그램을 종료합니다.".format(len(np.unique(ids))))
