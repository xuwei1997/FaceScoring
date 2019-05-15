# 冻结
# ResNet50模型训练网络

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from keras.applications import ResNet50
import matplotlib.pyplot as plt

# tf.test.gpu_device_name()
model_name = "Frozen"  # 模块命名，用于绘图时
train_epochs0 = 10  # 设置训练轮次


def show_history_mse(history0):  # 绘制mse图像
    plt.plot(history0.history['loss'])
    plt.plot(history0.history['val_loss'])
    plt.title(model_name + ' mse')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('drive/app/' + model_name + '_mse.jpg')
    plt.show()


def show_history_ce(history0):  # 绘制ce图像
    plt.plot(history0.history['ce'])
    plt.plot(history0.history['val_ce'])
    plt.title(model_name + ' ce')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('drive/app/' + model_name + '_ce.jpg')
    plt.show()


X = np.load('drive/app/X_data.npy')
Y = np.load('drive/app/Y_data.npy')
# 统一X和Y的数量级
X = X / 25
print("read the data")

# 切片，统一数量级
x_train = X[:5000]
y_train = Y[:5000]
x_test = X[5000:]
y_test = Y[5000:]
print("train data and test data")

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(220, 220, 3), pooling='avg')
model = Sequential()
model.add(resnet)
model.add(Dense(1))

model.layers[0].trainable = False  # 设置ResNet50不可训练

# print(resnet.summary())
print(model.summary())

print("compile")
model.compile(loss='mean_squared_error', optimizer=Adam())

print("fit")
Hist = model.fit(x_train, y_train, epochs=train_epochs0, batch_size=64, validation_data=(x_test, y_test))
print(Hist.history)
show_history_mse(Hist)
# show_history_ce(Hist)

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
# print(loss_and_metrics)

del X
del Y
del x_train
del y_train
del x_test
del y_test
# model.save('drive/app/my_model.h5')
