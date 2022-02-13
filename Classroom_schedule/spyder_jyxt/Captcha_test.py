import requests
import numpy as np
import cv2 as cv
from CNN_pred import model_init, show_img, image_process, pred


url_img = 'http://域名/verifycode.servlet?t=0.5466662795746771'
weight_path = '148train_acc00968888test_acc00980000.pth'
json_path = 'label_decoder_cnn.json'

image = requests.get(url_img).content
image = cv.imdecode(np.array(bytearray(image), dtype='uint8'), cv.IMREAD_UNCHANGED)
model, label_decoder = model_init(weight_path, json_path)

img_pro = image_process(image)
print(img_pro.shape)
captcha_pred = pred(model, img_pro, label_decoder)
show_img(image, ' ', captcha_pred)

