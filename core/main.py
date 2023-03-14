from core.predict import predict,pre_process
from redis_conn.redis_conn import Cache

# 处理完保存到服务器本地临时的目录下
def c_main(path,model,ext,conn:Cache):
    '''
    保存图片、返回图片名和图片信息
    '''
    image_data = pre_process(path)
    image_info = predict(image_data,model,ext,conn)
    return image_data[1]+'.'+ext, image_info # 返回图片名、图片信息

if __name__ == '__main__':
    pass