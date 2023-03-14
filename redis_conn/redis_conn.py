'''
实现redis的连接、插入图片、取图片
'''

import redis
import pickle
import base64
from PIL import Image
import numpy as np
import sys
import cv2
import io

class Cache:
    # 设置默认的redis服务ip和密码
    def __init__(self,host='127.0.0.1',password=''):
        pool = redis.ConnectionPool(host=host,password=password)
        self.conn = redis.Redis(connection_pool=pool)
        print('Redis连接成功!')

    def set(self,key,value):
        self.conn.set(key,value)

    def get(self,key):
        return self.conn.get(key)

    def hset(self,name,key,value):
        self.conn.hset(name,key,value)

    def hget(self,name,key):
        return self.conn.hget(name,key)

    def insertImage(self, img_path):
        """
        将图片路径的图片以base64格式插入redis
        """
        img = Image.open(img_path)
        arr = np.asarray(img)
        imageBytes = Cache._base64_encode_image(arr)  # 转化为了str
        b = pickle.dumps(imageBytes)
        self.conn.set(img_path,b)   # key：图片路径 value：str

    def insertImage_cv(self,img,filename,shape):
        w = shape[0]
        h = shape[1]
        imageBytes = Cache._opencv2base64(img)
        b = pickle.dumps(imageBytes)
        # self.conn.set(filename, b)  # key：图片路径 value：str
        self.conn.hset(filename,'w',w)
        self.conn.hset(filename,'h', h)
        self.conn.hset(filename,'img',b)
        self.conn.expire(filename,300)   # 设置过期时间

    def getImage(self,key):
        """
        获取图片：Img格式
        """
        w = int(self.conn.hget(key,'w'))
        h = int(self.conn.hget(key, 'h'))
        img = self.conn.hget(key, 'img')
        result = pickle.loads(img)
        image_arr = Cache._base64_decode_image(result, np.uint8, (h,w))
        image = Image.fromarray(image_arr)
        return image

    def image_to_base64(self, image):
        """
        输入为PIL读取的图片，输出为base64格式
        """
        byte_data = io.BytesIO()  # 创建一个字节流管道
        image.save(byte_data, format="JPEG")  # 将图片数据存入字节流管道
        byte_data = byte_data.getvalue()  # 从字节流管道中获取二进制
        base64_str = base64.b64encode(byte_data).decode("ascii")  # 二进制转base64
        return base64_str

    def _opencv2base64(img):
        """
        工具函数：
        opencv转Image转base64格式
        """
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))    # Image
        arr = np.asarray(image) # 数组
        imageBytes = Cache._base64_encode_image(arr)  # 转化为了str
        return imageBytes

    def return_img_stream(self,img_local_path):
        """
        工具函数:
        获取本地图片流
        :param img_local_path:文件单张图片的本地绝对路径
        :return: 图片流
        """
        img_stream = ''
        with open(img_local_path, 'rb+') as img_f:
            img_stream = img_f.read()
            img_stream = base64.b64encode(img_stream).decode()
        return img_stream

    def _base64_encode_image(arr):
        # base64 encode the input NumPy array
        return base64.b64encode(arr).decode("utf-8")

    def _base64_decode_image(result, dtype, shape):
        if sys.version_info.major == 3:
            result = bytes(result, encoding="utf-8")
        result = np.frombuffer(base64.decodebytes(result), dtype=dtype)
        result = result.reshape(shape[1], shape[0], -1)
        return result

    def exist(self,key):
        """
        判断key是否存在
        """
        return self.conn.exists(key)

    def insert_info(self,img_key,img_info:dict):
        img_key = img_key + '_info'
        for key,value in img_info.items():
            new_value = ','.join(value)
            self.conn.hset(img_key,key,new_value)
        self.conn.expire(img_key,300)    # 设置过期时间

    def get_img_info(self,img_key):
        """
        获取图片字典信息
        """
        dic = self.conn.hgetall(img_key)
        new_dic={}
        for key,val in dic.items():
            new_dic[str(key,encoding='utf-8')] = (str(val,encoding='utf-8')).split(',')
        return new_dic

if __name__ == '__main__':
    dic = {'person':['3*2','0.7'],'car':['4*4','0.85']}
    # for key, value in dic.items():
    #     dic[key] = ','.join(value)
    # print(dic)
    conn = Cache()
    conn.insert_info('test',dic)
    new_dic = conn.get_img_info('test_info')
    print(new_dic)
