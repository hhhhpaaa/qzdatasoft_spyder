import os
import cv2 as cv
import json
from collections import Counter
import h5py
import numpy as np


def dataset(path_pro):
    '''
    输入：处理后的图片路径

    图片二值化
    处理标签，进行编码
    将所有数据整合为矩阵

    返回：图片矩阵 标签矩阵
    '''
    img_data_list = []
    img_label_list = []
    files = os.listdir(path_pro)
    label_count = Counter()

    for i in files:
        img = cv.imread(os.path.join(path_pro, i), -1)
        _, img_binary = cv.threshold(img, 177, 255, cv.THRESH_BINARY)
        label_count.update(i[:4])
        img_data_list.append(img_binary.astype(np.float32))
        img_label_list.append(i[:4])

    label_encoder = label_encode(label_count)
    encoder_label_list = []
    for label_str in img_label_list:
        label_ = [label_encoder[i] for i in label_str]
        encoder_label_list.append(label_)

    img_data = np.array(img_data_list)
    img_label = np.array(encoder_label_list)

    print("img_data.shape:", img_data.shape)
    print("img_label.shape:", img_label.shape)

    return img_data, img_label


def label_encode(label_count):
    '''
    输入：Counter对象

    保存标签编码解码字典

    返回：标签编码字典
    '''
    str_cap = label_count.keys()
    label_encoder = dict(zip(list(str_cap), list(range(len(str_cap)))))
    label_decoder = dict(zip(list(range(len(str_cap))), list(str_cap)))
    save_dict(label_encoder, label_decoder)

    return label_encoder


def save_dict(label_encoder, label_decoder):
    label_encoder = json.dumps(label_encoder)
    with open('label_encoder_cnn.json', 'w') as json_file:
        json_file.write(label_encoder)

    label_decoder = json.dumps(label_decoder)
    with open('label_decoder_cnn.json', 'w') as json_file:
        json_file.write(label_decoder)
    print("write json success")


if __name__ == '__main__':
    path_pro = './img_pro/'
    img_data, img_label = dataset(path_pro)
    with h5py.File('train_images_pro.h5', 'w') as hf:
        hf.create_dataset("images_data", data=img_data)
        hf.create_dataset("images_label", data=img_label)
