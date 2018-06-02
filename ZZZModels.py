import os
import numpy as np

import torch
import torch.nn as nn

from resnet import resnet18

class ResNet18(nn.Module):

    def __init__(self, num_classes, isTrained):
    
        super(ResNet18, self).__init__()
        
        self.resnet18 = resnet18(pretrained=isTrained)

        kernelCount = self.resnet18.fc.in_features
        #512
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])
        '''
        for i in range(14):
            s = 'Classifier_' + str(i)
            setattr(self, s, nn.Sequential(nn.Linear(512,1),nn.Sigmoid()))
        '''
        self.fc = nn.Sequential(nn.Linear(kernelCount, num_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)
        '''
        fun = eval('self.Classifier_0')
        
        y = fun(x)
        for i in range(13):
            fun = eval('self.Classifier_'+str(i+1))
            y = torch.cat([y,fun(x)],1)
        #x = self.fc(x)
        x = y
        '''
        x = self.fc(x)
        
        return x

