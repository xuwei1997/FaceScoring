# 绘图！

import numpy as np
import matplotlib.pyplot as plt

model_name = "Finetuning2"


def make_image(loss, val_loss):  # 绘制mse图像
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title(model_name + ' mse')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(model_name + '_mse.jpg', api=200)
    plt.show()


if __name__ == "__main__":
    loss = [2.131331657, 2.199375189, 2.092778292, 1.967078746, 2.042238023, 2.062163583, 1.987574683,
            2.181317396, 1.947904652
            ]
    val_loss = [2.457896578, 2.302741749, 1.835303192, 1.799305452, 1.800193298, 3.162019506, 1.934037434,
                1.769503006, 2.831111441
                ]
    make_image(loss, val_loss)
