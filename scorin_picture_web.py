# web评分服务端
# coding:utf-8

from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
import os
import cv2
from keras.models import Sequential
from keras.models import load_model
import numpy as np
import time
from datetime import timedelta


def sc(imagePath,current):
    global model
    #imagePath=q.get()
    frame = cv2.imread(imagePath)
    sh = frame.shape
    print(sh)
    if sh[0] > 1079:  # 图片过大时，缩小图片
        frame = cv2.resize(frame, (int(sh[1] * 850 / sh[0]),850 ), interpolation=cv2.INTER_AREA)
        print(frame.shape)
        print("change size 1")
    elif sh[1] > 1920:
        frame = cv2.resize(frame, (1500, int(sh[0]*1500/sh[1])), interpolation=cv2.INTER_AREA)
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
        #j = model.predict((new_image / 25), batch_size=None, verbose=0, steps=None)
        #print (j)
        text = str(round(k[0][0],3))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 1)
    cv2.imwrite('static/images2/'+current+'.png',frame)
    #cv2.imshow('frame', frame)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("end!!")


# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        current = str(round(time.time()))
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        #upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        upload_path = os.path.join(basepath, 'static/images',current+".png")  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
        print(upload_path)

        # 使用Opencv转换一下图片格式和名称
        # img = cv2.imread(upload_path)
        # cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)

        sc(upload_path,current)

        return render_template('upload_ok.html', userinput=user_input, val1=current)

    return render_template('upload.html')


if __name__ == '__main__':
    model = Sequential()
    model = load_model('DenseNet121_2_model.h5')
    print("111111111111111111111111")
    test = np.zeros((1, 220, 220, 3))
    k = model.predict(test, batch_size=None, verbose=0, steps=None)
    print(k)
    app.run(host='localhost', port=8987, debug=False)