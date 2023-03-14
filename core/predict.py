import cv2 as cv
import os
import base64



def predict(dataset,model,ext,conn):
    '''
    dataset:数据
    model：模型
    ext：图片后缀
    预测图片，并保存
    '''
    global img_y
    x = dataset[0].replace('\\','/')    # 路径标准化
    filename = dataset[1]   # 文件名
    x = cv.imread(x) # 读取图片
    img_y,image_info = model.detect(x)  # 图片和图片信息
    shape = img_y.shape[:2]
    conn.insertImage_cv(img_y,filename,shape)   # 插入图片
    conn.insert_info(filename,image_info)       # 插入信息
    return image_info

def pre_process(data_path):
    '''
        返回图片路径和图片名
    '''
    file_name = os.path.split(data_path)[1].split('.')[0]
    return data_path, file_name
