import cv2
import numpy as np
from tensorflow.keras.models import load_model

emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Load model
emotion_classifier = load_model('emotion_model.h5')

def preprocess_face(face_img, face_cascade, gray_img):
    face_rect = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    
    if len(face_rect) > 0:
        (x, y, w, h) = face_rect[0]
        face_img = face_img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
    
    return face_img

def classify_emotion(face_img):
    global emotion_classifier
    
    predictions = emotion_classifier.predict(face_img)[0]
    max_index = np.argmax(predictions)
    
    return emotion_labels[max_index]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_img = gray[y:y+h, x:x+w]
        face_img = preprocess_face(face_img, face_cascade, gray)
        if face_img is not None:
            emotion = classify_emotion(face_img)
            cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()