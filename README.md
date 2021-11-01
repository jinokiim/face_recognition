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



2. 마스크 착용여부를 알려주는 프로그램


## FacialRecognition  


## 정보

김진호 – [@깃허브주소](https://github.com/jinokiim) – wlsghrla94@gmail.com
