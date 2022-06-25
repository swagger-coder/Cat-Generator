# Cat-News Title&Abstract Generator

## 环境
创建虚拟环境venv
- python 3.7
- streamlit 1.10.0
- transformers 3.0.2
- pytorch 1.11.0


```
pip install requirements.txt
```
#### pytorch无法下载
- 将`.\whl`下whl文件放到`venv\Lib\sit-packages`中
- 转到`venv\Lib\sit-packages`目录，`pip`安装对应`whl`文件即可

## 项目结构
- Cat-Text.py   主页面
- pages\Cat-File.py 多页面
- output_dir    模型文件
- static 静态文件
- utils 工具包
- vocab\vocab.txt   词汇表

## 运行
```
streamlit run Cat-Text.py
## 指定端口
streamlit run Cat-Text.py --server.port 80`
```

## 内网穿透
- 部署nginx，修改安装目录中的`conf\nginx.conf`
```
server {
    listen 80;
    server_name localhost;
    location / {
        proxy_pass http://0.0.0.0:8502/;    # streamlit server端口
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $http_host;
        proxy_redirect off;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;
    
    location = /50x.html {
        root /usr/share/nginx/html;
    }
 }
```
- 下载[natapp](https://natapp.cn/) ，根据[教程](https://natapp.cn/article/natapp_newbie) 配置，注意端口与nginx一致(如8502)
- `C:\Users\username\.streamlit`创建`config.toml`文件，并添加配置
```
[server]
port=8502 # change port number. By default streamlit uses 8501 port
headless=true # This will eliminate automatically open browser
[browser] # This ip and port will show in command prompt
serverAddress = "192.168.0.108" # Put your Local IP or Domain Name
serverPort = 8502
```
