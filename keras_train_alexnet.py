# 各个不同网络的冻结与微调
# 冻结和微调
# AlexNet 网络

from keras.models import Sequential,Model
from keras.optimizers import Adam
import numpy as np
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import GlobalAveragePooling2D, Input
import matplotlib.pyplot as plt
import gc

# tf.test.gpu_device_name()
model_name = "AlexNet"  # 选择使用哪种模型,ResNet50, VGG19, InceptionV3, MobileNet,NASNetMobile,DenseNet121
train_epochs0 = 20  # 设置冻结训练轮次
train_epochs1 = 15  # 设置微调训练轮次
ad = 0.0001  # 微调时学习率 ，冻结时默认0.001


def AlexNet(num_classses=1000):
    model = Sequential()

    model.add(ZeroPadding2D((2, 2), input_shape=(220, 220, 3)))
    model.add(Convolution2D(64, (11, 11), strides=(4, 4), activation='relu',name='1'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2),name='2'))

    model.add(ZeroPadding2D((2, 2),name='3'))
    model.add(Convolution2D(192, (5, 5), activation='relu',name='4'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2),name='5'))

    model.add(ZeroPadding2D((1, 1),name='6'))
    model.add(Convolution2D(384, (3, 3), activation='relu',name='7'))
    model.add(ZeroPadding2D((1, 1),name='8'))
    model.add(Convolution2D(256, (3, 3), activation='relu',name='9'))
    model.add(ZeroPadding2D((1, 1),name='10'))
    model.add(Convolution2D(256, (3, 3), activation='relu',name='11'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2),name='12'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu',name='15'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classses, activation='softmax',name='16'))

    return model


def show_history_mse2(history0, history1):  # 绘制mse图像
    plt.plot(history0.history['loss'] + history1.history['loss'])
    plt.plot(history0.history['val_loss'] + history1.history['val_loss'])
    plt.title(model_name + ' mse')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('drive/app/' + model_name + '_mse.jpg', dpi=200)
    plt.show()
    # 输出loss和val_loss
    print(history0.history['loss'] + history1.history['loss'])
    print(history0.history['val_loss'] + history1.history['val_loss'])


def prepare_data():  # 取数据
    print("prepare data")
    X = np.load('drive/app/X_data.npy')
    Y = np.load('drive/app/Y_data.npy')
    # 统一X和Y的数量级
    X = X / 25
    # 切片，统一数量级
    x_train = X[:5000]
    y_train = Y[:5000]
    x_test = X[5000:]
    y_test = Y[5000:]
    del X
    del Y
    c = gc.collect()  # 内存回收
    print(c)
    return (x_train, y_train, x_test, y_test)


def change_model(model0):  # 选择模型
    model = AlexNet(num_classses=1000)
    print(model)
    # model.save_weights('alexnet_weights.h5')
    model.load_weights('drive/app/alexnet_weights_2.h5',by_name=True)
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = prepare_data()
    transfer_model = change_model(model_name)
    model = Sequential()
    model.add(transfer_model)
    model.add(Dense(512, activation='relu',name="bbb"))
    model.add(Dense(1, name="aaa"))

    # Inp = Input((220, 220, 3))
    # x = transfer_model(Inp)
    # y = Dense(1, name="aaa")(x)
    # model = Model(inputs=Inp, outputs=y)

    # 冻结------------------------------------------
    print("Frozen!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # 设置transfer_model不可训练
    model.layers[0].trainable = False
    # print(transfer_model.summary())
    print(model.summary())

    print("compile")
    model.compile(loss='mean_squared_error', optimizer=Adam())

    print("fit")
    Hist = model.fit(x_train, y_train, epochs=train_epochs0, batch_size=64, validation_data=(x_test, y_test))

    model.save_weights('drive/app/weight.h5')
    print(Hist.history)

    # 微调---------------------------------------------
    print("Finetuning!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # 设置transfer_model可训练
    model2 = Sequential()
    model2.add(transfer_model)
    model2.add(Dense(512, activation='relu', name="bbb"))
    model2.add(Dense(1, name="aaa"))
    model2.load_weights('drive/app/weight.h5', by_name=True)
    for layer in model2.layers:
        layer.trainable = True

    print(model2.summary())

    print("compile")
    model2.compile(loss='mean_squared_error', optimizer=Adam(lr=ad))

    print("fit")
    Hist2 = model2.fit(x_train, y_train, epochs=train_epochs1, batch_size=64, validation_data=(x_test, y_test))
    print(Hist2.history)

    # 输出图像---------------------------------------------
    show_history_mse2(Hist, Hist2)

    model2.save('drive/app/' + model_name + '_model.h5')
    del x_train
    del y_train
    del x_test
    del y_test
    c = gc.collect()  # 内存回收
    print(c)
