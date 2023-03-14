from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
from datetime import timedelta
import time

from processor import detect_yolov5
from core import main
from redis_conn.redis_conn import Cache     # redis连接对象
import base64
import pickle
import io
# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg', 'JPEG'])




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        img_key = f.filename.split('.')[0]
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(basepath, 'static', f.filename)
        f.save(upload_path)  # 临时保存图片(可以考虑删除掉,防止占用空间)
        origin = conn.return_img_stream(upload_path)  # 原图base64

        if conn.exist(img_key):     # 如果图片存在的话,直接取图片
            t1 = time.time()
            detect_img = conn.getImage(img_key)  # 检测图片
            detect_img_base64 = conn.image_to_base64(detect_img)  # base64检测图片
            img_info = conn.get_img_info(img_key+'_info')
            t2 = time.time()
            print("调用redis缓存,耗时：{:.3}s\n".format((t2-t1)),f.filename)
            print(img_info)

        else:
            t1 = time.time()
            pid, img_info = main.c_main(upload_path,model,f.filename.rsplit('.')[1],conn=conn)  # 预测

            detect_img = conn.getImage(img_key)   # 检测图片
            detect_img_base64 = conn.image_to_base64(detect_img)    # base64检测图片
            t2 = time.time()
            print("cuda检测耗时：{:.3}s".format(t2-t1))
            print(pid)  # 图片名字
            print(img_info)  # 打印类别、类别宽高、置信度

        os.remove(upload_path)  # 删除图片

        return render_template('test.html',img_info=img_info, origin = origin, detect_img = detect_img_base64)
    return render_template('send.html')

if __name__ == '__main__':
    model = detect_yolov5.Detector()
    conn = Cache()
    app.run(host='0.0.0.0', port=5000,debug=True)
