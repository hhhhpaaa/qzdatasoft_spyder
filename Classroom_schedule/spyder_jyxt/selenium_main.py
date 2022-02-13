import requests
import numpy as np
import cv2 as cv
from selenium import webdriver
from lxml import etree
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from CNN_pred import model_init, image_process, pred


class chrome_driver(object):
    '''
    该对象用于创建webdriver
    '''
    def __init__(self, url_, user_name, password, url_error):
        '''
        初始化webdriver, 默认以无头模式运行
        '''
        super(chrome_driver, self).__init__()
        option = Options()
        option.add_argument('--headless')
        option.add_argument('--disable-gpu')
        option.add_argument('window-size=1024,768')
        option.add_argument('--no-sandbox')
        self.driver = webdriver.Chrome(options=option)
        self.driver.get(url_)
        self.driver.implicitly_wait(30)

        self.driver.find_element(By.XPATH, '//*[@id="userAccount"]').send_keys(user_name)
        self.driver.find_element(By.XPATH, '//*[@id="userPassword"]').send_keys(password)

        self.url_error = url_error

    def cookie_(self, model, label_decoder):
        '''
        该函数用于解决登录验证码并获取cookies

        返回 cookies 登录后跳转的URL
        '''
        element = self.driver.find_element(By.XPATH, '//*[@id="SafeCodeImg"]')
        self.driver.find_element(By.XPATH, '//*[@id="RANDOMCODE"]').send_keys(self.captcha_pred(
            element, model, label_decoder))
        self.driver.find_element(By.XPATH, '//*[@id="btnSubmit"]').click()
        if self.driver.current_url == self.url_error:
            self.cookie_(model, label_decoder)

        return self.driver.current_url, self.driver.get_cookies()

    def captcha_pred(self, element, model, label_decoder):
        '''
        该函数用于识别验证码
        '''
        image = element.screenshot_as_png   # 将网页上验证码保存为图片
        image = cv.imdecode(np.array(bytearray(image), dtype='uint8'), cv.IMREAD_UNCHANGED)
        image = image[:, :65, :]  # 将图片空白处切分掉

        img_pro = image_process(image)
        captcha_pred = pred(model, img_pro, label_decoder)

        return captcha_pred

    def driver_close(self):

        self.driver.quit()


def response_class(url_, url_next, cookies, week_num):
    '''
    该函数用于获取课表页面的response

    创建session对象，加载cookies
    进行页面跳转

    返回 课表页面的response
    '''
    data = {'zc': week_num, 'xnxq01id': '2021-2022-2', 'sfFD': 1}
    session = requests.session()

    requests.utils.add_dict_to_cookiejar(session.cookies, {cookies[0]['name']: cookies[0]['value']})
    response = session.get(url_next)
    response.encoding = 'utf-8'

    url_class = url_ + etree.HTML(response.text).xpath('/html/body/div[5]/a[1]/@href')[0]
    response_class = session.get(url_class)
    response_class.encoding = 'utf-8'

    url_class_schedule = url_ + etree.HTML(response_class.text).xpath(
        '//*[@id="LeftMenu1_divChildMenu"]/ul[2]/li/a/@href')[0]
    response_class_schedule = session.post(url_class_schedule, data=data)
    response_class_schedule.encoding = 'utf-8'

    return response_class_schedule.text


def response_xpath(response_):
    '''
    该函数用于处理课表页面的response

    返回 课表字典
    '''
    dict_index = {"1、2节": 2, "3、4节": 3, "6、7节": 5, "8、9节": 6}
    name_xpath = '//*[@id="kbtable"]/tr[{}]/td/div[2]/text()'
    room_xpath = '//*[@id="kbtable"]/tr[{}]/td/div[2]/font[@title="教室"]/text()'
    week_xpath = '//*[@id="kbtable"]/tr[{}]/td/div[2]/font[@title="周次(节次)"]/text()'

    dict_all = {}
    for key, value in dict_index.items():
        name_list = etree.HTML(response_).xpath(name_xpath.format(value))
        room_list = etree.HTML(response_).xpath(room_xpath.format(value))
        week_list = etree.HTML(response_).xpath(week_xpath.format(value))
        dict_all[key] = [name_list, room_list, week_list]

    return dict_all


if __name__ == '__main__':

    url_ = "http://域名"  # 登录url
    url_error = 'http://域名/Logon.do?method=logon'  # 验证码错误跳转的url
    user_name = 'uername'  # 登录用户名
    password = 'password'  # 登录密码
    weight_path = '148train_acc00968888test_acc00980000.pth'
    json_path = 'label_decoder_cnn.json'
    week_num = 2  # 获取第二周的课表

    model, label_decoder = model_init(weight_path, json_path)

    driver_ = chrome_driver(url_, user_name, password, url_error)
    url_next, cookies = driver_.cookie_(model, label_decoder)
    driver_.driver_close()

    response_ = response_class(url_, url_next, cookies, week_num)
    dict_all = response_xpath(response_)

    print(dict_all)


