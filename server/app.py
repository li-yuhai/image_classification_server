
import base64
from PIL import Image
from io import BytesIO
import time


import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, Depends
from fastapi import FastAPI

from detector import Detector

app = FastAPI()

# 配置允许域名
origins = [
    "*"
]
# origins = ["*"]  # 允许所有的请求

#  配置允许域名列表、允许方法、请求头、cookie等
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 这里接受请求图片的base64编码信息,可以一次传递多张图片
'''
    {
      "data": [
        "<base64-encoded-image-data-1>",
        "<base64-encoded-image-data-2>",
        "<base64-encoded-image-data-3>",
        ...
      ]
    }
    返回的图片是base64编码, 直接发挥预测的结果
'''



# 定义全局变量
global_var = Detector()

def global_variable():
    # 初始化全局变量
    return global_var


@app.post('/predict')
async def predict( request: Request , detector: Detector = Depends(global_variable)):
    # 从请求中检索出Base64编码的图像数据
    json_data = await request.json()
    data = json_data['data']
    # 初始化一个空数组以存储每个图像的预测结果
    predictions = []
    # starttime = time.time()

    for item in data:
        img_name = item['name']
        image_data = item['img']
        # 解码Base64编码的图像数据并转换为NumPy数组
        decoded_data = base64.b64decode(image_data)
        image = Image.open(BytesIO(decoded_data))
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        # detector = PestDetector()
        # image_pred, predicted = detector.predict(image, img_name)
        res = detector.predict(img_name, image)  # 使用全局变量来做

        # 将当前图像的预测结果（包括预测的图像本身）添加到预测结果数组中
        predictions.append(res)

    # end = time.time()
    # print(end - starttime)
    return predictions


if __name__ == '__main__':
    uvicorn.run(app =app, host = '0.0.0.0', port=8000)


