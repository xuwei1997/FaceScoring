# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
from keras.models import Sequential
from keras.models import load_model
import numpy as np


def Modle(model0, test):
    k = model0.predict(test, batch_size=None, verbose=0, steps=None)
    print(k)
    return (k[0][0])


def Scoring(model, x, y, w, h):
    new_image = frame[y:y + h, x:x + w]
    new_image = cv2.resize(new_image, (220, 220), interpolation=cv2.INTER_CUBIC)
    new_image = np.array([new_image])  # (1,220,220,3)
    print(new_image.shape)
    # k=Modle(model,new_image)
    k = model.predict(new_image, batch_size=None, verbose=0, steps=None)
    print(k)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    cv2.putText(frame, k[0][0], (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 1)


if __name__ == '__main__':
    # Create the haar cascade
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    model = Sequential()
    model = load_model('my_model.h5')
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()  # 读帧,frame是帧

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        print("Found {0} faces!".format(len(faces)))
        print(frame.shape)
        print(faces)
        # Draw a rectangle around the faces
        # 该函数返回四个值：矩形的 x 和 y 坐标，以及它的高和宽
        for (x, y, w, h) in faces:
            # Scoring(model,x, y, w, h)
            new_image = frame[y:y + h, x:x + w]
            new_image = cv2.resize(new_image, (220, 220), interpolation=cv2.INTER_CUBIC)
            new_image = np.array([new_image])  # (1,220,220,3)
            print(new_image.shape)
            # k=Modle(model,new_image)
            k = model.predict(new_image, batch_size=None, verbose=0, steps=None)
            print(k)
            text = str(k[0][0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 1)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
