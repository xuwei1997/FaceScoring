# 冻结和微调
# ResNet50模型训练网络

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD
import numpy as np
from keras.applications import ResNet50
import matplotlib.pyplot as plt

# tf.test.gpu_device_name()
model_name = "FrozenAndFinetuning"  # 模块命名，用于绘图时
train_epochs0 = 2  # 设置冻结训练轮次
train_epochs1 = 2  # 设置微调训练轮次

def show_history_mse2(history0,history1):  # 绘制mse图像
    plt.plot(history0.history['loss']+history1.history['loss'])
    plt.plot(history0.history['val_loss']+history1.history['val_loss'])
    plt.title(model_name+' mse')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('drive/app/'+model_name+'_mse.jpg')
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
model.add(Dense(1,name="aaa"))

# 冻结------------------------------------------
print("Frozen!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# 设置ResNet50不可训练
model.layers[0].trainable = False
# print(resnet.summary())
print(model.summary())

print("compile")
model.compile(loss='mean_squared_error', optimizer=Adam())

print("fit")
Hist = model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_test, y_test))

model.save_weights('drive/app/weight.h5')
print(Hist.history)

# 微调---------------------------------------------
print("Finetuning!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# 设置ResNet50可训练

model2=Sequential()
model2.add(resnet)
model2.add(Dense(1,name="aaa"))
model2.load_weights('drive/app/weight.h5',by_name=True)
for layer in model2.layers:
    layer.trainable = True

print(model2.summary())

print("compile")
model2.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0005))

print("fit")
Hist2 = model2.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_test, y_test))
print(Hist2.history)

# 输出图像---------------------------------------------
show_history_mse2(Hist,Hist2)

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