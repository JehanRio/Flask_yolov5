import torch
import numpy as np
from models.experimental import attempt_load    # 用于加载模型权重文件并构建模型（可以构造普通模型或者集成模型）
from utils.general import non_max_suppression, scale_coords  # NMS、缩放图像坐标
from utils.torch_utils import select_device     # 选择模型训练的设备 并输出日志信息
from utils.augmentations import letterbox   # 自适应缩放图片
import cv2
from random import randint

class Detector():
    def __init__(self):
        self.img_size = 640
        self.threshold = 0.4
        self.max_frame = 160
        self.init_model()   # 初始化模型

    def init_model(self):
        self.weights = 'weights/person_car.pt'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights,device=self.device)
        model.to(self.device).eval()    # 开启评估模式
        model.half()    # 半精度:节省显存并加快推理速度 只有gpu才行
        self.m = model
        self.names = model.module.names if hasattr(model,'module') else model.names     # 获取类别名字符串列表
        self.colors = [(randint(0,255),randint(0,255),randint(0,255)) for _ in self.names]  # 对应框的颜色

    def preprocess(self,img):
        # 图片预处理
        img0 = img.copy()
        img = letterbox(img,new_shape=self.img_size)[0] # 缩放后的图片
        img = img[:,:,::-1].transpose(2,0,1)    # 改为rgb HWC->CHW bgr
        img = torch.from_numpy(img.copy()).to(self.device) # 转为numpy，并转换到设备上
        img = img.half()
        img /= 255.0    # 图像归一化
        if img.ndimension() == 3:   # 如果没有batch_size,在最前面添加一个轴
            img = img.unsqueeze(0)  # (1,3,384,640)

        return img0,img # 返回原图和预处理后的图片

    def plot_bboxs(self,image,bboxs,line_thickness=None):
        tl = line_thickness or round(0.002*(image.shape[0]+image.shape[1])/2)+1    # 线宽
        for (x1,y1,x2,y2,cls_id,conf) in bboxs:
            color = self.colors[self.names.index(cls_id)]   # 框的颜色
            c1,c2 = (x1,y1),(x2,y2) # 两个坐标
            cv2.rectangle(image,c1,c2,color,thickness=tl,lineType=cv2.LINE_AA)  # 抗锯齿类型，很顺滑看着
            tf = max(tl-1,1)    # font thickness
            t_size = cv2.getTextSize(cls_id,0,fontScale=tl/3,thickness=tf)[0]   # 获取一个文字的宽度和高度
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image,'{} {:.2f}'.format(cls_id,conf),(c1[0],c1[1]-2),0,tl/3,
                        [225,225,225],thickness=tf,lineType=cv2.LINE_AA) # 在图片上绘制文字
        return image

    def detect(self,im):

        im0,img = self.preprocess(im)

        pred = self.m(img,augment=False)[0] # 前向传播
        pred = pred.float()
        # 后处理
        pred = non_max_suppression(pred, self.threshold, 0.3) # 置信度阈值、IOU阈值

        pred_boxes = []
        image_info = {}
        count = 0
        for det in pred:    # pred是一个batch中的所有，遍历batch的每张图片
            if det is not None and len(det):
                # 调整预测框的坐标：基于resize+pad的图片坐标-->基于size图片的坐标，此时坐标为xyxy
                det[:,:4] = scale_coords(img.shape[2:],det[:,:4],im0.shape).round()

            for *x,conf,cls_id in det:
                lb = self.names[int(cls_id)]
                x1, y1 = int(x[0]), int(x[1])
                x2, y2 = int(x[2]), int(x[3])
                pred_boxes.append((x1,y1,x2,y2,lb,conf))
                count+=1
                key = '{}-{:02}'.format(lb,count)   # 类别-数目
                image_info[key] = ['{}×{}'.format(x2-x1, y2-y1), str(np.round(float(conf),3))] # 图片位置 置信度

        im = self.plot_bboxs(im, pred_boxes)

        return im, image_info   # 返回图片和图片信息