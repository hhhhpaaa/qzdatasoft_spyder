FROM python:3.8.12
EXPOSE 20001
WORKDIR /usr/src/app
COPY spyder_jyxt/ google-chrome.deb chromedriver_linux64.zip requirements.txt sources.list  ./
RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime &&\
    cp sources.list /etc/apt/sources.list  &&\
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 425956BB3E31DF51 &&\
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32  &&\
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 871920D1991BC93C  &&\
    apt-get update   &&\
    apt-get install libgdk-pixbuf2.0-common=2.40.0+dfsg-3ubuntu0.2  --allow-downgrades -y &&\
    apt-get install fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 libatspi2.0-0 libcups2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 libxcomposite1 libxdamage1 libxfixes3 libxkbcommon0 libxrandr2 xdg-utils -y &&\
    apt-get install -f -y &&\
    dpkg -i google-chrome.deb &&\
    apt-get install -f -y  &&\
    apt-get autoremove -y &&\
    unzip chromedriver_linux64.zip &&\
    mv chromedriver /usr/local/share/chromedriver &&\
    ln -s /usr/local/share/chromedriver /usr/bin/chromedriver
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
CMD ["python","./api_class_schedule.py"]
