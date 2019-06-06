# web评分服务端
# -*-coding:utf-8-*-
from flask import Flask, render_template, request
import os
import base64
import cv2
from keras.models import Sequential
from keras.models import load_model
import numpy as np
import time

def sc(imagePath, current):
    global model
    # imagePath=q.get()
    frame = cv2.imread(imagePath)
    sh = frame.shape
    print(sh)
    if sh[0] > 1079:  # 图片过大时，缩小图片
        frame = cv2.resize(frame, (int(sh[1] * 850 / sh[0]), 850), interpolation=cv2.INTER_AREA)
        print(frame.shape)
        print("change size 1")
    elif sh[1] > 1920:
        frame = cv2.resize(frame, (1500, int(sh[0] * 1500 / sh[1])), interpolation=cv2.INTER_AREA)
        print(frame.shape)
        print("change size 2")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    print("Found {0} faces!".format(len(faces)))
    print(frame.shape)
    print(faces)
    for (x, y, w, h) in faces:
        new_image = frame[y:y + h, x:x + w]
        new_image = cv2.resize(new_image, (220, 220), interpolation=cv2.INTER_CUBIC)
        new_image = np.array([new_image])  # (1,220,220,3)
        print(new_image.shape)
        # k=Modle(model,new_image)
        # 注意！此处一定要/25，统一数量级！与训练时的神经网络保持一致
        k = model.predict((new_image / 25), batch_size=None, verbose=0, steps=None)
        print(k)
        print("!!!!!")
        # j = model.predict((new_image / 25), batch_size=None, verbose=0, steps=None)
        # print (j)
        text = str(round(k[0][0], 3))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 1)
    cv2.imwrite('static/images2/' + current + '.png', frame)
    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("end!!")


app = Flask(__name__)
basepath = os.path.dirname(__file__)
html = '''<img src="data:image/png;base64,{}" style="width:100%;height:100%;"/>'''


@app.route('/upload', methods=['GET', 'POST'])  # 接受并存储文件
def up_file():
    if request.method == "POST":
        current = str(round(time.time()))
        # 保存图片
        # request.files['file'].save(os.path.join(basepath, 'static/images', current + ".png"))
        upload_path = os.path.join(basepath, 'static/images', current + ".png")
        request.files['file'].save(upload_path)
        # 处理图片
        sc(upload_path, current)
        # 发送图片
        return html.format(base64.b64encode(open("static/images2/" + current + ".png", 'rb').read()).decode())


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('upload.html')


if __name__ == "__main__":
    model = Sequential()
    model = load_model('DenseNet121_2_model.h5')
    print("111111111111111111111111")
    test = np.zeros((1, 220, 220, 3))
    k = model.predict(test, batch_size=None, verbose=0, steps=None)
    print(k)
    app.run(host='localhost', port=5000, debug=False)