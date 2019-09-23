# -*- coding: utf-8 -*-
# @Time    : 2018/4/24 10:04
# @Author  : zhoujun

import torch
import torch.nn.functional as F
import cv2
import os
import numpy as np
import time
import sys
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from ZZZModels import ResNet18

class Pytorch_model:
    def __init__(self, model_path, img_shape, img_channel=3, transCrop= 224, gpu_id=None, classes_txt=None):
        self.gpu_id = gpu_id
        self.img_shape = img_shape
        self.img_channel = img_channel
        self.transCrop = transCrop
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % (self.gpu_id))
        else:
            self.device = torch.device("cpu")

        if self.gpu_id is not None and isinstance(self.gpu_id, int):
            self.use_gpu = True
        else:
            self.use_gpu = False

        net = ResNet18(14, False)
        #modelCheckpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

        #if not self.use_gpu:
        #net.load_state_dict(modelCheckpoint['state_dict'])
        self.net1 = nn.Sequential(*list(net.resnet18.children())[:-1])  
        self.net2 = nn.Sequential(list(net.resnet18.children())[-1])
        self.net3 = list(net.children())[-1]

        #else:
            #net.load_state_dict(modelCheckpoint['state_dict']).cuda()
        self.net1.eval()
        self.net2.eval()
        self.net3.eval()
        self.weight = list(net.parameters())[-2]
        self.weight = torch.clamp(self.weight,min=0)
        #print(self.weights.shape)
        self.ww1 = list(self.net1.parameters())[-1]
        self.ww3 = list(self.net1.parameters())[-3]
        self.ww4 = list(self.net1.parameters())[-4]
        self.ww5 = list(self.net1.parameters())[-5]
        self.ww6 = list(self.net1.parameters())[-6]
        #print(self.net1)

        if classes_txt is not None:
            with open(classes_txt, 'r') as f:
                self.idx2label = dict(line.strip().split(' ')
                                      for line in f if line)
        else:
            self.idx2label = None

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize((self.transCrop, self.transCrop)))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)

    def predict(self, image_path, is_numpy=False, topk=1):
        if len(self.img_shape) not in [2, 3] or self.img_channel not in [1, 3]:
            raise NotImplementedError
        with torch.no_grad():
            if not is_numpy and self.img_channel in [1, 3]:  # read image
                #img = cv2.imread(img, 0 if self.img_channel == 1 else 1)
                img = Image.open(image_path).convert('RGB')
                
            imageData = self.transformSequence(img)
            imageData = imageData.unsqueeze_(0)
            #print(imageData)
            #print(imageData)
            '''
            img = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))
            if len(img.shape) == 2 and self.img_channel == 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and self.img_channel == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        

            img = img.reshape(
                [self.img_shape[0], self.img_shape[1], self.img_channel])
            tensor = transforms.ToTensor()(img)
            tensor = tensor.unsqueeze_(0)
            tensor = Variable(tensor)

            tensor = tensor.to(self.device)
            outputs = F.softmax(self.net(tensor), dim=1)
            result = torch.topk(outputs.data[0], k=topk)
            '''
        
            input = imageData
            
            output = self.net1(input)
            realo = self.net2(output)
            realo = realo.view(realo.size(0), -1)
            #print(realo)
            realo = self.net3(realo)

            realo = torch.squeeze(realo)
            #print(realo)

        #output = output.resize(1,14,1,1)
        #print(realo)

        #index = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        #        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        index = [ '0Atelectasis', '1Cardiomegaly', '2Effusion', '3Infiltration', '4Mass', '5Nodule', '6Pneumonia',
                '7Pneumothorax', '8Consolidation', '9Edema', '10Emphysema', '11Fibrosis', '12Pleural_Thickening', '13Hernia']
        
        prob = realo.numpy().tolist()
        print(prob)
        print(self.weight.shape)
        a = realo.reshape((1,14))
        b = self.weight
        weights = torch.mm(a, b)
        weights = weights.reshape(512)
        print(weights.shape)

        #---- Generate heatmap
        #print(len(self.weights))
        heatmap = None
        for i in range (0, len(weights)):
            map = output[0,i,:,:]
            if i == 0: heatmap = weights[i] * map
            else: heatmap += weights[i] * map
        
        #---- Blend original and heatmap 
        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(image_path, 1)
        img_size = 336
        img = cv2.resize(imgOriginal, (img_size, img_size))
        
        #cam = npHeatmap / np.max(npHeatmap)
        #cam = cv2.resize(cam, (self.transCrop, self.transCrop))
        cam = npHeatmap
        cam = cv2.resize(cam, (img_size, img_size), cv2.INTER_CUBIC)
        cam = cam / np.max(cam)
        np.where(cam > 0.5, cam*cam-cam/2+0.5, cam)
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
        img = heatmap * 0.3 + img * 0.6
            
        cv2.imwrite(image_path+'.heat.jpg', img)
        heatmap_path = image_path+'.heat.jpg'

        '''
        if self.device != "cpu":
            index = result[1].cpu().numpy().tolist()
            prob = result[0].cpu().numpy().tolist()
        else:
            index = result[1].numpy().tolist()
            prob = result[0].numpy().tolist()
        '''
        if self.idx2label is not None:
            label = []
            for idx in index:
                label.append(self.idx2label[str(idx)])
            result = zip(label, prob)
        else:
            result = zip(index, prob)

        result = [(index, prob) for (index, prob) in result if prob>=0.1]
        result = sorted(result, key=lambda x : x[1], reverse=True)
        return result, os.path.split(heatmap_path)[1]


if __name__ == '__main__':
    model_path = 'ckpt.pth.tar'
    gpu_id = None
    model = Pytorch_model(model_path=model_path, img_shape=[224, 224], img_channel=3, gpu_id=gpu_id)
    image_path = '666.png'
    result,_ = model.predict(image_path=image_path, is_numpy=False,topk=3)
    for label, prob in result:
        print('label:%s,probability:%.4f'%(label, prob))
