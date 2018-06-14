# -*- coding: utf-8 -*-
# @Time    : 2018/5/2 14:36
# @Author  : Davy
import flask
from flask_uploads import UploadSet, IMAGES, configure_uploads, ALL
from flask import request, Flask, redirect, url_for, render_template
import time
import cv2
import numpy as np
import config
from werkzeug.utils import secure_filename
from model3 import Pytorch_model
import os
import psutil

app = Flask(__name__)
app.config.from_object(config)
photos = UploadSet('PHOTO')
configure_uploads(app, photos)


@app.route('/chexnet')
def index():
    return render_template('demo.html')


@app.route('/demo', methods=['POST', 'GET'])
def upload():
    print(psutil.Process(os.getpid()).memory_info().rss)
    if request.method == 'POST':
        img = request.files['img'].filename
        #topk = request.form['topk']
        topk = 1
        img = secure_filename(img)
        new_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '_' + img
        filename = photos.save(request.files['img'], name=new_name)
        #print(filename)
        #print(request.files['img'])
        data, heatmap_path = predict_img(photos.path(filename), is_numpy=False,topk=int(topk))
        #img_path = photos.url(filename)
        #print(img_path)
        #print(heatmap_path)
        #img_path = heatmap_path.split('/')[-1]
        img_path = photos.url(heatmap_path)
        print(img_path)
        return flask.jsonify({"result":data,"img_path":img_path})
    else:
        img_path = None
        result = []
    return render_template('upload.html', img_path=img_path, result=result)

'''
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = {'state': False}
    if request.method == 'POST':
        img = request.files['image'].read()
        try:
            topk = request.form['topk']
        except:
            topk = 1
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, flags=1)
        data = predict_img(img, is_numpy=True, topk=topk)
    return flask.jsonify(data)
'''


def predict_img(img, is_numpy=False, topk=1):
    data = dict()
    start = time.time()
    result, img_path = model.predict(img, is_numpy=is_numpy, topk=int(topk))
    cost_time = time.time() - start
    data['predictions'] = list()
    for label, prob in result:
        m_predict = {'label': label, 'probability': ("%.4f" % prob)}
        data['predictions'].append(m_predict)
    data['state'] = True
    data['time'] = cost_time
    return data, img_path


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    model_path = 'ckpt.pth.tar'
    gpu_id = None
    model = Pytorch_model(model_path=model_path, img_shape=[
        224, 224], img_channel=3, gpu_id=gpu_id)
    app.run(host='0.0.0.0', port=5000, debug=True)
    print(psutil.Process(os.getpid()).memory_info().rss)
