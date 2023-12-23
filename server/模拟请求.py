

import base64

import cv2
import numpy as np
from PIL import Image
from io import BytesIO


# 读取图像文件
image_path = "100015.jpg"  # 替换成实际的图像文件路径
with open(image_path, "rb") as image_file:
    # 将图像数据读取为字节流
    image_data = image_file.read()

# 将字节流编码为 Base64
base64_encoded = base64.b64encode(image_data).decode('utf-8')

# 打印 Base64 编码
# print("Base64 Encoded:", base64_encoded)



import requests
import base64

# 构造要发送的数据，包含多个 Base64 编码的图像数据
# data_to_send = {
#     "data": [
#         "<base64-encoded-image-data-1>",
#         "<base64-encoded-image-data-2>",
#         "<base64-encoded-image-data-3>",
#         # 添加更多图像数据
#     ]
# }

data_to_send = {
    "data": [
        {'name':'test.jpg', 'img': base64_encoded}
        # 添加更多图像数据
    ]
}


# 定义接口的 URL
# url = "http://59.110.152.138:8000/predict"  # 替换成实际的服务器地址
# url = "http://127.0.0.1:8000/predict"  # 替换成实际的服务器地址
url = "http://59.110.152.138:5555/predict"  # 替换成实际的服务器地址

# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',"Content-Type": "application/json"}


# 发送 POST 请求
# response = requests.post(url, json=data_to_send, headers=headers)
response = requests.post(url, json=data_to_send, timeout=30)

# 检查请求是否成功
if response.status_code == 200:
    # 解析响应结果
    predictions = response.json()
    print("Predictions:", predictions)
else:
    print("Error:", response.status_code, response.text)



print(response)
# for item in predictions:
#     image_data = item['image']
#     image_name = item['imgname']
#
#     decoded_data = base64.b64decode(image_data)
#     image = Image.open(BytesIO(decoded_data))
#     if image.mode == 'RGBA':
#         image = image.convert('RGB')
#     image.save('test_img/' + image_name)

    # decoded_im0_bgr = cv2.imdecode(np.frombuffer(decoded_data, np.uint8), cv2.IMREAD_COLOR)
    # cv2.imwrite('test_img/' + image_name, decoded_im0_bgr)
