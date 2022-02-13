from flask import Flask, request
import json
from gevent import pywsgi
from selenium_main import chrome_driver, response_class, response_xpath
from CNN_pred import model_init, image_process, pred

url_ = "http://域名"  # 登录url
url_error = 'http://域名/Logon.do?method=logon'  # 验证码错误跳转的url
user_name = 'uername'   # 登录用户名
password = 'password'  # 登录密码
weight_path = '148train_acc00968888test_acc00980000.pth'
json_path = 'label_decoder_cnn.json'

model, label_decoder = model_init(weight_path, json_path)  # 初始化模型

app = Flask(__name__)


@app.route("/api_data", methods=["POST"])
def check():
    '''
    该函数封装了post请求
    '''
    return_dict = {'return_code': '200', 'return_info': '处理成功'}
    if not request.get_data():
        return_dict['return_code'] = '504'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    get_Data = request.get_data()
    get_Data = json.loads(get_Data)

    if get_Data.get('password') == '001231':
        week = get_Data.get('week')
        dict_room = get_data(week)
        return_dict['dict'] = dict_room

        return json.dumps(return_dict, ensure_ascii=False)

    else:
        return_dict['password'] = 'error'
        return json.dumps(return_dict, ensure_ascii=False)


def get_data(week):
    '''
    该函数用于登录网站，并获取课表数据

    创建webdriver对象
    获取下一个URL和cookies
    关闭webdriver

    获取课表网页的response
    解析数据，获取课表数据字典

    返回 课表字典
    '''
    driver_ = chrome_driver(url_, user_name, password, url_error)
    url_next, cookies = driver_.cookie_(model, label_decoder)
    driver_.driver_close()

    response_ = response_class(url_, url_next, cookies, week)
    dict_all = response_xpath(response_)

    return dict_all


if __name__ == "__main__":

    server = pywsgi.WSGIServer(('0.0.0.0', 20001), app)
    server.serve_forever()
