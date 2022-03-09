################################
'''
    基于keras框架检测Alexnet网络
    环境： tf123
'''
################################

import numpy as np
import utils
import cv2
from keras import backend as K
from Alexnet_keras import Alexnet

# 将图像数据中的通道数提前, n*h*w*c -> c*n*h*w
K.image_data_format() == 'channels_first'

# 开始验证
if __name__ == "__main__":
    # 读取网络结构和权重文件
    model = Alexnet()
    model.load_weights("./logs/last1.h5")

    # 读取待检测的图像并进行预处理
    img = cv2.imread("./test.jpg")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor, axis = 0)
    img_resize = utils.resize_image(img_nor, (224, 224))

    # 直接输出结果
    print('the answer is: ', utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)