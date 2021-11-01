# face_recognition

### 1. 간단한 얼굴인식, 코와 입 검출을 하는 파이썬 프로그램  

### 2. 웹캠에 접속하여 사진(ID)을 등록하고 학습하여 해당 ID에 맞는 얼굴을 검출하는 파이썬 프로그램


## 사용

Python & OpenCV



## 내용

face_recognition  
* Cascades(검출을 위한 분류기생성을 위한 파일)  
   * haarcascade_frontalface_default.xml    
   * haarcascade_mcs_mouth.xml    
   * haarcascade_mcs_nose.xml    
   * etc    
  
* FaceDetection    

   * 01_facedetection.py : 얼굴 인식    
   * 02_nosedetection.py : 코 인식  
   * 03_noseandmouthdetection.py : 마스크 작용 여부를 위한 얼굴, 코, 입 인식    
     
* FacialRecognition  
   * 01_check_cam.py  : 캠 확인  
   * 02_face_detection.py : 얼굴 인식 확인    
   * 03_face_makeID.py : 얼굴에 맞는 ID 생성    
   * 04_face_training.py : 이미지 학습    
   * 05_face_recognition.py  : ID를 통한 얼굴인식  
   * 06_face_recognition_tkinter.py  : 간단한 GUI  
  
          
          
## FaceDetection    
1. 얼굴사진 등록 후 사진과 같은 얼굴 검출
#### 실행장면
<img width="1257" alt="1" src="https://user-images.githubusercontent.com/88222461/139668488-3e7d1530-08b6-4c2e-8613-a42b65ecc047.png">


2. 마스크 착용여부를 알려주는 프로그램
#### 실행장면
<img width="1257" alt="2" src="https://user-images.githubusercontent.com/88222461/139668564-6b3b8cf6-7140-4728-a329-14c30b50a9e9.png">
<img width="1257" alt="3" src="https://user-images.githubusercontent.com/88222461/139668586-53f99cf1-d829-4bc2-b7db-2ba5a2e29b95.png">
<img width="1257" alt="4" src="https://user-images.githubusercontent.com/88222461/139668592-8a795aef-e76e-47fb-b0a2-5048ffc18fd2.png">


## FacialRecognition  
1. 30장의 사진을 촬영후 폴더에 저장한 모습
#### 실행장면
<img width="479" alt="5" src="https://user-images.githubusercontent.com/88222461/139668596-904e70ff-1464-40a7-a120-3e2957c83000.png">

2. 30장의 사진을 이용해 데이터를 학습하고 얼굴 검출한 모습
#### 실행장면
<img width="1261" alt="6" src="https://user-images.githubusercontent.com/88222461/139668597-f90a336c-6466-4a75-a98a-76963fca524d.png">


## 정보

김진호 – [깃허브주소](https://github.com/jinokiim) – wlsghrla94@gmail.com
