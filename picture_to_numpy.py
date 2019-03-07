# 将照片转换成矩阵保存
import numpy as np
import cv2
import os
import random
from read_excel import get_data

X = []
Y = []
listdir = os.listdir("E:\\daxue\\graduation\\face")  # 读取文件名
data = get_data('data', 'data_sheet')

random.shuffle(listdir)#打乱读取照片的顺序！

for i in listdir:
    print(i)
    imagePath = "E:\\daxue\\graduation\\face\\" + i
    image = cv2.imread(imagePath)
    X.append(image)
    Y.append(data[i])

X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
np.save('X_data', X)
np.save('Y_data', Y)
