import numpy as np
import cv2
import pickle
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
upper_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
   og_labels = pickle.load(f)
   labels = {v:k for k,v in og_labels.items()}


cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, 1.1,4)


    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h,x:x+w]

        id_, conf = recognizer.predict(roi_gray)

        if conf>=45: # and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            cv2.putText(img, name, (x,y), font, 1, (255, 255,255), 2, cv2.LINE_AA)

            """for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                cv2.putText(img, 'Face', (x, y), font, 2, (255, 0, 0), 5)

            smile = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=35,
                minSize=(25, 25),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(roi_color, (sh, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
                cv2.putText(img,'smile',  (x + sx, y + sy), 1, 1, (0, 255, 0), 1)"""
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(img, 'eye' , (x + ex, y + ey), 1, 1, (0, 255, 0), 1)
            cv2.putText(img, 'Number of Faces : ' + str(len(faces)), (40, 40), font, 1, (255, 0, 0), 2)

            img_item = "elon.jpg"
            cv2.imwrite(img_item,roi_color)

        cv2.rectangle(img, (x,y), (x+w, y+h),(255, 0, 0), 2)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        subitems= smile_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in subitems:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('img', img)


    if cv2.waitKey(20) & 0xFF == ord('q'):
             break

cap.release()
