from flask import Flask, render_template, request, jsonify, send_file
from redis_conn.redis_conn import Cache     # redis连接对象
#
app = Flask(__name__)
conn = Cache()
from PIL import Image


def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    img_stream = ''
    with open(img_local_path, 'rb+') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream


@app.route('/')
def hello_world():
    img_path = 'E:\\图片\\1.jpg'
    img_stream = return_img_stream(img_path)
    return render_template('index.html',
                           img_stream=img_stream)


# 注意：在img标签中的src一定要按照 data:;base64,{{img_stream}} 的形式添加，否则显示不出图片。

if __name__ == '__main__':
    app.run(debug=True, port=8080)
