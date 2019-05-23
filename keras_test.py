#测试网络，输出loss图像
from keras.models import Sequential
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model_name = "ResNet50"

X = np.load('drive/app/X_data.npy')
Y = np.load('drive/app/Y_data.npy')
X = X / 25

x_test = X[5000:]
y_test = Y[5000:]

model=Sequential()
model=load_model('drive/app/' + model_name + '_model.h5')

y_pre=model.predict(x_test, batch_size=None, verbose=0, steps=None)

plt.title(model_name+" test")
plt.xlabel("y_test")
plt.ylabel("y_pre")
plt.scatter(y_test,y_pre)
#plt.plot(x_test,x_pre,"ob")
plt.savefig('drive/app/'+model_name+'_test.jpg', dpi=200)
plt.show()