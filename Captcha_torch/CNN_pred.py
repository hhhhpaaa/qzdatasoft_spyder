import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from CNN_NET import DNNNet
from collections import Counter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def image_process(image):
    '''
    输入：图片矩阵 三通道

    转换为灰度图片，二值化处理
    去除黑边
    将面积小于20的噪点去除

    返回：图片矩阵 单通道
    '''
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, img_binary = cv.threshold(img_gray, 177, 255, cv.THRESH_BINARY)

    img_binary[:2, :] = 255
    img_binary[-2:, :] = 255
    img_binary[:, :2] = 255
    img_binary[:, -2:] = 255

    _, contours, _ = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area <= 20:
            cv_contours.append(contour)
        else:
            continue
    img_pro = cv.fillPoly(img_binary, cv_contours, 255)

    return img_pro


def pred(model, img_captcha, label_decoder):
    '''
    输入：模型 图片矩阵 标签解码字典

    图片增加维度(1,1,22,62)
    重叠矩阵(10,1,22,62)，进行十次预测
    矩阵归一化
    十次预测中取最多的预测标签
    标签解码

    输出：预测标签
    '''
    if img_captcha.shape != (22, 62):
        img_captcha = cv.resize(img_captcha, (62, 22))
    image_matrix = np.asarray(img_captcha).astype(np.float32)
    image_matrix = np.expand_dims(np.expand_dims(image_matrix, axis=0), axis=0)
    image_matrix = np.repeat(image_matrix, 10, axis=0) / 255.
    image_matrix = torch.from_numpy(image_matrix)

    model.eval()
    image_matrix = image_matrix.to(device)
    with torch.no_grad():
        output_ = model(image_matrix)
    _, label_pred = output_.max(-1)

    index_list = []
    for i in range(label_pred.shape[1]):
        (index_, _) = Counter(label_pred[:, i].tolist()).most_common()[0]
        index_list.append(index_)

    captcha_pred = "".join([label_decoder[str(i)] for i in index_list])

    return captcha_pred


def model_init(weight_path, json_path):
    '''
    输入：权重路径 解码JSON路径

    创建模型，加载权重
    导入标签解码字典

    输入：模型 标签解码字典
    '''
    model = DNNNet(22 * 62, 512, 256, 40)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)

    with open(json_path, 'r', encoding='utf8') as fp:
        label_decoder = json.load(fp)

    return model, label_decoder


def show_img(img_captcha, captcha_true, captcha_pred):
    plt.imshow(img_captcha, cmap=plt.cm.gray)
    plt.title("true_label:{} pred_label:{}".format(captcha_true, captcha_pred))
    plt.show()


if __name__ == '__main__':

    weight_path = './checkpoint_CNN/148train_acc00968888test_acc00980000.pth'
    json_path = 'label_decoder_cnn.json'
    image_path = 'c:Users/c2793/Desktop/verifycode (2).jfif'
    captcha_true = 'nvc2'

    image = cv.imread(image_path, 1)
    img_pro = image_process(image)
    model, label_decoder = model_init(weight_path, json_path)

    captcha_pred = pred(model, img_pro, label_decoder)
    show_img(image, captcha_true, captcha_pred)