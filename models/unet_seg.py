# ------------------------------------------------------------------------------
# Copyright (c) SenseTime
# Written by heyinan@sensetime.com
# ------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

__all__ = ['xception_seg']
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


class Xception_seg(nn.Module):
    def __init__(self, num_classes=1000, dropout=None, bn_group=None, bn_sync_stats=True, use_sync_bn=True, **kwargs):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        global BN
        
        BN = nn.BatchNorm2d

        print(bn_group)

        bypass_bn_weight_list = []
        self.inplanes = 64
        
        super(Xception_seg, self).__init__()
        self.num_classes = num_classes
        self.dropout=dropout

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
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
        
        self.seg = nn.Conv2d(2048, 1,
                             kernel_size=1, bias=True)
        
        self.upsample = nn.Upsample(size=256)

        if not self.dropout:
            self.fc = nn.Linear(2048, num_classes)
        else:
            print('Using dropout', dropout)
            self.model.fc = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(2048, num_classes)
            )


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

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def forward(self, input, output_embed=False):
        x = self.features(input)
        x = self.relu(x)
        
        seg = self.seg(x)
        seg = self.upsample(seg).squeeze(1)
        

        embed = F.adaptive_avg_pool2d(x, (1, 1))
        embed = embed.view(embed.size(0), -1)
        logit = self.fc(embed)
        if output_embed:
            return seg, logit, embed
        else:
            return seg, logit

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

def load_pretrain(model, pretrain_path):
    # Load model in torch 0.4+
    state_dict = torch.load(pretrain_path)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            if len(weights.shape)<=2:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    for k in missing_keys:
        print('caution: missing keys from checkpoint {}'.format(k))
    pretrain(model, state_dict)

def pretrain(model, state_dict):
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
                print("But don't worry about it. Continue pretraining.")

def xception_seg(pretrain_path, **kyargs):
    model = Xception_seg(**kyargs)
    load_pretrain(model, pretrain_path)
    return model


if __name__=="__main__":
    model = xception_seg(num_classes=1)
    image = torch.randn(1,3,256,256).cuda()
    model = model.cuda()
    ret1,ret2 = model(image)
    print(ret1.shape)
    pdb.set_trace()
