import json
import os

import torch
from django.http import HttpResponse
from django.shortcuts import render

from detect.detect1 import Detect

d = Detect()


def hello(request):
    return render(request, 'index.html')


def upload(request):
    data = {}
    if request.method == "POST":
        fp = request.FILES.get("file")
        # fp 获取到的上传文件对象
        if fp:
            path = os.path.join('static/', 'img/1.jpg')  # 上传文件本地保存路径， image是static文件夹下专门存放图片的文件夹
            # fp.name #文件名
            # yield = fp.chunks() # 流式获取文件内容
            # fp.read() # 直接读取文件内容
            with open(path, 'wb') as f:
                bytes = fp.read()
                f.write(bytes)
            print("小文件上传完毕")
            # path = os.path.abspath(path)

            data['code'] = 1
            with torch.no_grad():
                json_data = d.detect_binary(bytes)
                data['data'] = json_data
        else:
            data['code'] = 0
    return HttpResponse(json.dumps(data), content_type="application/json,charset=utf-8")
