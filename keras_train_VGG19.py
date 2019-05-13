#VGG19模型训练网络
#import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.utils import plot_model
#from keras.applications import transfer_model50
from keras.applications import VGG19
import matplotlib.pyplot as plt

#tf.test.gpu_device_name()

def show_history(history0):  # 绘制图像
    plt.plot(history0.history['loss'])
    #plt.plot(history0.history['val_acc'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('drive/app/VGG19.jpg')
    plt.show()


X = np.load('drive/app/X_data.npy')
Y = np.load('drive/app/Y_data.npy')
#X = np.swapaxes(X, 1, 3)
#Y = to_categorical(Y, num_classes=None)
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

transfer_model = VGG19(include_top=False,weights='imagenet',input_shape=(220,220,3), pooling='avg')
model = Sequential()
model.add(transfer_model)
model.add(Dense(1))

#model.layers[0].trainable = False#设置transfer_model50不可训练

#print(transfer_model.summary())
print(model.summary())

print("compile")
#!!metrics是评价函数，详见文档
model.compile(loss='mean_squared_error', optimizer=Adam(),metrics=['accuracy','mae','crossentropy'])

print("fit")
Hist = model.fit(x_train, y_train, epochs=2, batch_size=64)
print(Hist.history)
show_history(Hist)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)

#model.save('drive/app/my_model.h5')