#ResNet50模型训练网络

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from keras.applications import ResNet50
import matplotlib.pyplot as plt

#tf.test.gpu_device_name()

def show_history(history0):  # 绘制图像
    plt.plot(history0.history['loss'])
    #plt.plot(history0.history['val_acc'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('drive/app/1.jpg')
    plt.show()


X = np.load('drive/app/X_data.npy')
Y = np.load('drive/app/Y_data.npy')

X = X / 25
# print(X[0][0])

print("read the data")

# print(X)
# print(Y)


# 切片，统一数量级
x_train = X[:5000]
y_train = Y[:5000]
x_test = X[5000:]
y_test = Y[5000:]
print("train data and test data")

resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(220,220,3), pooling='avg')
model = Sequential()
model.add(resnet)
model.add(Dense(1))

#model.layers[0].trainable = False#设置ResNet50不可训练

print(resnet.summary())
print(model.summary())

print("compile")
model.compile(loss='mean_squared_error', optimizer=Adam(),metrics=['accuracy','crossentropy'],validation_data=(x_test,y_test))

print("fit")
Hist = model.fit(x_train, y_train, epochs=3, batch_size=64)
print(Hist.history)
show_history(Hist)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)

model.save('drive/app/my_model.h5')