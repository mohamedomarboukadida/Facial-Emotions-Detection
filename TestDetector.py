
import cv2
import numpy as np
from time import sleep
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

face_classifier = cv2.CascadeClassifier(r'E:\Users\Desktop\Facial_Emotions_Detection\haarcascade_frontalface_default.xml')
model =load_model(r'E:\Users\Desktop\Facial_Emotions_Detection\model.h5')

emotion_classes = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

video_capture = cv2.VideoCapture(0)


while True:
    _, frame = video_capture.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        face_region = gray[y:y+h,x:x+w]
        face_region = cv2.resize(face_region,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([face_region])!=0:
            face_array = face_region.astype('float')/255.0
            face_array = img_to_array(face_array)
            face_array = np.expand_dims(face_array,axis=0)

            predictions = model.predict(face_array)[0]
            emotion =emotion_classes[predictions.argmax()]
            text_position = (x,y)
            cv2.putText(frame,emotion,text_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces Detected',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()