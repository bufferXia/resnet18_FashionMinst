import numpy as np
import os

'''
test code
# image_path = r"H:\Dataset\FashionMNIST/raw"
# train_image,train_label = load_Fashion_Minst(image_path,"train")
# test_image,test_label = load_Fashion_Minst(image_path,"t10k")
'''

def load_Fashion_Minst(path,train_test="train"):

    #读取图像文件名
    image_file_name = '{}-images-idx3-ubyte'.format(train_test)
    label_file_name = '{}-labels-idx1-ubyte'.format(train_test)
    datalenth = 60000 if train_test=="train" else 10000
    fd = open(os.path.join(path, image_file_name))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((datalenth, 28, 28, 1)).astype(np.uint8)  #跳过文件头

    fd = open(os.path.join(path, label_file_name))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((datalenth)).astype(np.uint8)

    #标签数据转化为onehot编码

    return trX,trY

