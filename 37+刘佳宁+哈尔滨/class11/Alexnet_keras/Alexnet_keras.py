################################
'''
    基于keras框架实现Alexnet
    环境： tf123
'''
################################

from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam

# 建立Alexnet网络模型
def Alexnet(input_shape = (224,224,3), output_shape = 2):

    # 利用Sequential叠加网络层建立模型
    model = Sequential()

    # 第一层：48个卷积核, 卷积核大小11*11, 步长4, 填充0
    model.add(
        Conv2D(
            filters = 48,
            kernel_size = (11,11),
            strides = (4,4),
            padding = 'valid',
            input_shape = input_shape,
            activation = 'relu'
        )
    )

    # BN + Maxpooling
    model.add(BatchNormalization())
    model.add(
        MaxPooling2D(
            pool_size = (3,3),
            strides = (2,2),
            padding = 'valid'
        )
    )

    # 第二层：128个卷积核, 卷积核大小5*5, 步长1, 填充2
    model.add(
        Conv2D(
            filters = 128,
            kernel_size = (5,5),
            strides = (1,1),
            padding = 'same',
            activation = 'relu'
        )
    )

    # BN + Maxpooling
    model.add(BatchNormalization())
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 第三层：192个卷积核, 卷积核大小3*3, 步长1, 填充2
    model.add(
        Conv2D(
            filters = 192,
            kernel_size = (3,3),
            strides = (1,1),
            padding = 'same',
            activation = 'relu'
        )
    )

    # 第四层：192个卷积核, 卷积核大小3*3, 步长1, 填充2
    model.add(
        Conv2D(
            filters = 192,
            kernel_size = (3,3),
            strides = (1,1),
            padding = 'same',
            activation = 'relu'
        )
    )

    # 第五层：128个卷积核, 卷积核大小3*3, 步长1, 填充2
    model.add(
        Conv2D(
            filters = 128,
            kernel_size = (3,3),
            strides = (1,1),
            padding = 'same',
            activation = 'relu'
        )
    )

    # BN + Maxpooling
    model.add(BatchNormalization())
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 展平操作flatten层
    model.add(Flatten())

    # 第六层：全连接层
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    # 第七层：全连接层
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    # 第八层：全连接层
    model.add(Dense(output_shape, activation='softmax'))

    return model