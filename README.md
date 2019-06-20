##人脸颜值打分系统

基于华南理工大学SCUT-FBP5500_v2数据集训练

1.read_picture 读取照片，识别出人脸，保存为220*220jpg图片。

2.read_excel 读取打分数据

3.picture_to_numpy 将照片保存为numpy数组

4.keras_trai_Frozen 冻结

5.keras_trai_Finetuning 冻结

6.keras_trai_FroAndFin 冻结后微调

7.keras_train_2 探究不同源模型的性能

8.scoring_picture 为照片中人脸打分

9.keras_test 测试神经网络

10.live 为摄像头实时视频中人脸打分