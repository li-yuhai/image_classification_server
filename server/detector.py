from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import json
from torchvision import transforms

from model import shufflenet_v2_x1_0


class Detector():
    def __init__(self ):
        # read class_indict
        num_classes = 12
        json_path = './class_indices.json'
        model_path = 'shufflenetv2_x1_622.pth'

        # 处理json
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r") as f:
            self.class_indict = json.load(f)

        # device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # model
        self.model = shufflenet_v2_x1_0(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        #
        print("模型初始化成功过！！！")
    # image的数据类型是 io
    def predict(self, image_name, image):
        img = self.data_transform(image)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # predict class
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        # predict_res = "class: {}   prob: {:.3}".format(self.class_indict[str(predict_cla)],
        #                                         predict[predict_cla].numpy())

        return { "image_name": image_name, "class": self.class_indict[str(predict_cla)], "prob": round(predict[predict_cla].numpy().item(), 2)}


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    detector = Detector()
    image = Image.open('100015.jpg')
    res = detector.predict( 'test', image)
    print(res)


