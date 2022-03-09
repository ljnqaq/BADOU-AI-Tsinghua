# 第11周作业：

1. alexnet完整实现（训练+推理）
2. vgg实现（选做）
3. resnet50推理实现


Alexnet网络：
1. Alexnet_kera.py:
    按串联顺序叠加卷积层和全连接层构成Alexnet网络
2. Alexnet_train.py:
    对输入图片进行预处理后输入到网络模型，经过前向传播，后向传播，优化进行训练
    其中训练好的权重文件放置在logs中last1.h5文件中(因为我的网络模型是完整的Alexnet，所以需要重新训练，该权重文件仅对此网络有效)
3. Alexnet_test.py:
    将待验证的图片输入到已经训练好的模型中，检验网络效果

Resnet50网络：
1. ResNet50_keras.py:
    分别建立identity_block和conv_block两个模块，进而构建ResNet网络
    权重文件于resnet50_weights_tf_dim_ordering_tf_kernels.h5中