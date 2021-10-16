import cv2
import face_recognition
 
# 이미지 불러오기
image_to_be_matched = face_recognition.load_image_file('./img/forproject/jihnotestface.png')
name = "Jinho Kim"
 
# 이미지 인코딩
image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]
print(image_to_be_matched_encoded)
 

cap = cv2.VideoCapture(0)
 
if not cap.isOpened():  # 카메라가 없을때
    print("no camera")
    exit()
    

# 프로그램 실행 
while cap.isOpened():
    status, frame = cap.read()
 
    if not status:
        print("Could not read frame")
        exit()
 
    # CNN 기반 얼굴 검출
    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
 
    for face_location in face_locations:
        # 얼굴 위치 확인
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        # 얼굴 검출 및 이름 확인
        try:
            face_encoded = face_recognition.face_encodings(face_image)[0]
            result = face_recognition.compare_faces([image_to_be_matched_encoded], face_encoded, 0.5)
 
            if result[0] == True:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                Y = top - 10 if top - 10 > 10 else top + 10
                text = name
                cv2.putText(frame, text, (left, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except:
            pass
    cv2.imshow("detect", frame)
 
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
