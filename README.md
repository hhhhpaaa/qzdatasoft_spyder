# qzdatasoft_spyder
## 一个爬取强智教务系统课程数据的demo

### 思路

使用selenium模拟登录网站，对验证码进行识别。登录网站后获取cookie，携带cookie请求网站获取数据。

### 验证码识别模型

1. #### 数据处理

一共爬取了500条验证码图片，进行手动打码。使用opencv、numpy库对验证码进行处理，包括：二值化、去除边框以及消除噪点。将数据归一化后存储为h5文件，作为模型训练数据集。对标签进行编码，一共有"zxcvbnm123"10种字母。

原验证码图片：

![](https://github.com/hhhhpaaa/qzdatasoft_spyder/blob/master/Captcha_torch/img_label/1b2v.jpg)     ![1bc3](https://github.com/hhhhpaaa/qzdatasoft_spyder/blob/master/Captcha_torch/img_label/1bc3.jpg)     ![1bnb](https://github.com/hhhhpaaa/qzdatasoft_spyder/blob/master/Captcha_torch/img_label/1bnb.jpg)     ![1bx2](https://github.com/hhhhpaaa/qzdatasoft_spyder/blob/master/Captcha_torch/img_label/1bx2.jpg)

处理后的验证码图片：

![](https://github.com/hhhhpaaa/qzdatasoft_spyder/blob/master/Captcha_torch/img_pro/1b2v.jpg)     ![1bc3](https://github.com/hhhhpaaa/qzdatasoft_spyder/blob/master/Captcha_torch/img_pro/1bc3.jpg)     ![1bnb](https://github.com/hhhhpaaa/qzdatasoft_spyder/blob/master/Captcha_torch/img_pro/1bnb.jpg)     ![1bx2](https://github.com/hhhhpaaa/qzdatasoft_spyder/blob/master/Captcha_torch/img_pro/1bx2.jpg)

2.  #### 模型

使用pytorch框架搭建训练模型，模型包括CNN和DNN。实际使用的模型为DNN，DNN在训练集和测试集上时准确率可以达到98％。本人在训练CNN时，其准确率不如DNN，可能是模型和训练参数设置有问题。

损失函数采用nn.CrossEntropyLoss()，图片维度为(22, 62,1)，模型输入维度为(batch_size, 1, 22, 62)，输出维度为(batch_size,4,10)。训练4个分类器，每个分类器负责区分10种字母。

![acc](https://github.com/hhhhpaaa/qzdatasoft_spyder/blob/master/Captcha_torch/acc.png)

![loss](https://github.com/hhhhpaaa/qzdatasoft_spyder/blob/master/Captcha_torch/loss.png)

3.   #### 预测

每一个分类器对图片进行10次预测，取最多的预测值作为此图片的预测标签。

### 主程序

使用selenium模拟登录网站，借助DNN模型识别验证码，登录成功后获取cookie，携带cookie请求网站，得到response，使用XPATH解析html，得到课程数据。使用flask框架，将程序封装为接口，以便后续调用。

请求示例：

```python
import requests
import json

url = 'http://server_IP:20001/api_data'
datafarm = {"password":"001231","week":"1"}
response = requests.post(url,data=json.dumps(datafarm))
print(response.text)
```

返回数据：

```
'{"return_code": "200", "return_info": "处理成功", "dict": {"1、2节": [["\xa0", "\xa0", "EDA技术与应用", "\xa0", "\xa0", "\xa0", "\xa0"], ["D104"], ["1-8(周)(限选)"]], "3、4节": [["\xa0", "\xa0", "传感器原理与应用", "\xa0", "传感器原理与应用", "\xa0", "\xa0"], ["B204", "B201"], ["1-10(周)(必修)", "1-10(周)(必修)"]], "6、7节": [["EDA技术与应用", "单片机与接口技术", "\xa0", "\xa0", "\xa0", "\xa0", "\xa0"], ["D104", "信息楼526"], ["1-8(周)(限选)", "1-12(周)(必修)"]], "8、9节": [["电子信息工程专业科研训练与课程论文（设计）", "单片机与接口技术", "电子信息工程专业科研训练与课程论文（设计）", "\xa0", "\xa0", "\xa0", "\xa0"], ["A209", "信息楼526", "A209"], ["1-4(周)(必修)", "1-12(周)(必修)", "1-4(周)(必修)"]]}}'
```

### 服务部署

使用docker将服务部署在云端linux服务器。docker镜像由Classroom_schedule文件夹中的Dockerfile文件构建，所需基础镜像为[python 3.8.12](https://hub.docker.com/_/python)，其余依赖文件均在Classroom_schedule文件夹中。本人使用python 3.8.12构建的镜像较大，另外可以基于[chromedriver](https://hub.docker.com/r/spryker/chromedriver)镜像构建。

