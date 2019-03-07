##人脸颜值打分系统

基于华南理工大学SCUT-FBP5500_v2数据集训练

1.read_picture 读取照片，识别出人脸，保存为220*220jpg图片。

2.read_excel 读取打分数据

3.picture_to_numpy 将照片保存为numpy数组

4.keras_train 训练神经网络

5.keras_train_ResNet50 利用ResNet50模型进行迁移学习

5.scoring_picture 为照片中人脸打分

6.live 为摄像头实时视频中人脸打分