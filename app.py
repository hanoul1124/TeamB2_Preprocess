from flask import Flask, request
from preprocess import resizing_human
from create_model import *

app = Flask(__name__)


# case1: model을 사전에 생성하지 않은 방식
# @app.route('/file', methods=['POST'])
# def file_upload():
#     file = request.files['file']
#     resized_img = resizing_human(file)
#     resized_img.save('./uploaded_file.jpg')
#     return 'upload completed!'


# case2: 사전에 모델을 생성하는 방식
net = create_model('Unet_human')
net.eval()


@app.route('/file', methods=['POST'])
def file_upload():
    file = request.files['file']
    resized_img = resizing_human(file, model=net, temp_size=384)
    resized_img.save('./uploaded_file.jpg')
    return 'upload completed!'


if __name__ == '__main__':
    app.run()


