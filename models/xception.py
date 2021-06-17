# ------------------------------------------------------------------------------
# Copyright (c) SenseTime
# Written by Joey Fang (fangzheng@sensetime.com)
# ------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

__all__ = ['xception']


BN = None
class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = BN(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(BN(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(BN(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(BN(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, in_channels = 3, num_classes=1000, bn_group_size=1, 
                 bn_group=None, bn_sync_stats=True,feature_visible=False, 
                 dropout=0, return_feature_idx=None, **kwargs):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        global BN

        BN = nn.BatchNorm2d

        bypass_bn_weight_list = []
        self.inplanes = 64

        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.return_feature_idx = return_feature_idx
        self.feature_visible = feature_visible

        self.conv1 = nn.Conv2d(in_channels, 32, 3,2, 0, bias=False)
        self.bn1 = BN(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = BN(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = BN(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = BN(2048)

        self.fc = nn.Linear(2048, num_classes)
        self.drop = None
        if dropout > 0:
            self.drop = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif (isinstance(m, SyncBatchNorm2d)
                  or isinstance(m, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if bypass_last_bn:
            for param in bypass_bn_weight_list:
                param.data.zero_()
            print('bypass {} bn.weight in BottleneckBlocks'.format(len(bypass_bn_weight_list)))

    def att_feature(self, feature):
        sum_feature = F.relu(torch.sum(feature, dim=1))
        sum_feature = sum_feature / (torch.max(sum_feature)+ 1e-9)
        return sum_feature

    def features(self, input):
        features = []
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        features.append(x)

        x_b1 = self.block1(x)
        features.append(x_b1)
        x_b2 = self.block2(x_b1)
        features.append(x_b2)
        x_b3 = self.block3(x_b2)
        features.append(x_b3)
        x_b4 = self.block4(x_b3)
        features.append(x_b4)
        x_b5 = self.block5(x_b4)
        features.append(x_b5)
        x_b6 = self.block6(x_b5)
        features.append(x_b6)
        x_b7 = self.block7(x_b6)
        features.append(x_b7)
        x_b8 = self.block8(x_b7)
        features.append(x_b8)
        x_b9 = self.block9(x_b8)
        features.append(x_b9)
        x_b10 = self.block10(x_b9)
        features.append(x_b10)
        x_b11 = self.block11(x_b10)
        features.append(x_b11)
        x_b12 = self.block12(x_b11)
        features.append(x_b12)

        x = self.conv3(x_b12)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x, features

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if self.drop is not None:
            x = self.drop(x)
        out = self.fc(x)
        return out, x

    def forward(self, input):
        x, features = self.features(input)
        logit, embedding = self.logits(x)
        features.append(embedding)
        selected_feature = None
        if self.return_feature_idx is not None:
            selected_feature = [features[i] for i in self.return_feature_idx]
            if self.feature_visible:
                selected_feature = [self.att_feature(feature) for feature in selected_feature]
            return logit, selected_feature
        else:
            return logit

def get_model_size(model):
    result = 0
    for key,value in model.state_dict().items():
        s = 1
        for item in value.size():
            s *= item
        result += s
        print(key)
    result *= 4
    return result

def xception(pretrain_path=None, **kwargs):
    model = Xception(**kwargs)
    # print(kwargs)
    if pretrain_path != None:
        state_dict = torch.load(pretrain_path)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except:
                    print('While copying the parameter named {}, '
                          'whose dimensions in the model are {} and '
                          'whose dimensions in the checkpoint are {}.'
                          .format(name, own_state[name].size(), param.size()))
    return model

if __name__=="__main__":
    model = xception(num_classes=2)
    model.eval()
    print(get_model_size(model)/1024/1024)
