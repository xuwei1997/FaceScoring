# 给照片打分
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
    cv2.putText(frame, str(k[0][0]), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 1)


if __name__ == '__main__':
    model = Sequential()
    model = load_model('my_model.h5')
    imagePath = "004.jpg"
    frame = cv2.imread(imagePath)
    sh=frame.shape
    if sh[0]>1920 or sh[1]>1920:#图片过大时，缩小图片
        frame = cv2.resize(frame, (1500 ,int(sh[0]*1500/sh[1])), interpolation=cv2.INTER_AREA)
        print(frame.shape)
        print("change size")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    print("Found {0} faces!".format(len(faces)))
    print(frame.shape)
    print(faces)
    for (x, y, w, h) in faces:
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
    cv2.imshow('frame', frame)
    cv2.waitKey(0)