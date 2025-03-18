from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ActorNetwork(nn.Module):
    def __init__(self, npy_path, num_actions=4):
        super(ActorNetwork, self).__init__()

        self.data_dict = np.load(npy_path, allow_pickle=True, encoding='latin1').item() if npy_path else None

        self.conv1 = nn.Conv2d(3, self.data_dict['conv1'][0].shape[3], kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, self.data_dict['conv2'][0].shape[3], kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.data_dict['conv3'][0].shape[3], kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(self.conv3.out_channels, self.data_dict['conv4'][0].shape[3], kernel_size=3, stride=1)

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            for param in layer.parameters():
                param.requires_grad = False

        self.lrn = nn.LocalResponseNorm(size=5)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def extract_features(self, x):
        x = x - 128.0
        x = self.pool(self.lrn(F.relu(self.conv1(x))))
        x = self.pool(self.lrn(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x.view(x.size(0), -1)

    def forward(self, rgb_g, rgb_l):
        x_g = self.extract_features(rgb_g)
        x_l = self.extract_features(rgb_l)

        x = torch.cat((x_g, x_l), dim=1)

        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        resnet_model = resnet18(weights=True)
        self.features = nn.Sequential(*list(resnet_model.children())[:-2])
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.fc1 = nn.Linear(512 * 4 * 4  , 512)  
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x,output_layer ):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        
        if (output_layer =="feature"):
            return x
        

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat((x), dim=1)
        x = self.fc3(x)

        if output_layer=="fc3":
            return x
        elif output_layer=="fc3_softmax":
            return F.softmax(x)
        
        return x

import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
   
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x


class Actor(nn.Module):
    def __init__(self, model_path=None):
        super(Actor, self).__init__()
        if model_path is not None:
            self.params_values_list = self._import_model(model_path)
        else:
            self.params_names_list = None
            self.params_values_list = None

        self.conv1_l = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.relu1_l = nn.ReLU()
        self.LRNl1 = LRN()
        self.pool1_l = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2_l = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        self.relu2_l = nn.ReLU()
        self.LRNl2 = LRN()
        self.pool2_l = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3_l = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.relu3_l = nn.ReLU()

        self.conv4_l = nn.Conv2d(512, 512, kernel_size=3, stride=2)
        self.relu4_l = nn.ReLU()

        self.conv1_g = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.relu1_g = nn.ReLU()
        self.LRNg1 = LRN()
        self.pool1_g = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2_g = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        self.relu2_g = nn.ReLU()
        self.LRNg2 = LRN()
        self.pool2_g = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3_g = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.relu3_g = nn.ReLU()

        self.conv4_g = nn.Conv2d(512, 512, kernel_size=3, stride=2)
        self.relu4_g = nn.ReLU()

        self.fc1 = nn.Linear(1024, 512)
        self.relu5 = nn.ReLU()

        self.fc2 = nn.Linear(512, 3)
        self.out = nn.Tanh()

        self.init_weight()


    def init_weight(self):
        self.conv1_l.weight.data = torch.from_numpy(np.transpose(self.params_values_list['conv1'][0], [3, 2, 0, 1]))
        self.conv2_l.weight.data = torch.from_numpy(np.transpose(self.params_values_list['conv2'][0], [3, 2, 0, 1]))
        self.conv3_l.weight.data = torch.from_numpy(np.transpose(self.params_values_list['conv3'][0], [3, 2, 0, 1]))
        self.conv4_l.weight.data = torch.from_numpy(np.transpose(self.params_values_list['conv4'][0], [3, 2, 0, 1]))

        self.conv1_l.bias.data = torch.from_numpy(np.squeeze(self.params_values_list['conv1'][1]))
        self.conv2_l.bias.data = torch.from_numpy(np.squeeze(self.params_values_list['conv2'][1]))
        self.conv3_l.bias.data = torch.from_numpy(np.squeeze(self.params_values_list['conv3'][1]))
        self.conv4_l.bias.data = torch.from_numpy(np.squeeze(self.params_values_list['conv4'][1]))

        self.conv1_g.weight.data = torch.from_numpy(np.transpose(self.params_values_list['conv1'][0], [3, 2, 0, 1]))
        self.conv2_g.weight.data = torch.from_numpy(np.transpose(self.params_values_list['conv2'][0], [3, 2, 0, 1]))
        self.conv3_g.weight.data = torch.from_numpy(np.transpose(self.params_values_list['conv3'][0], [3, 2, 0, 1]))
        self.conv4_g.weight.data = torch.from_numpy(np.transpose(self.params_values_list['conv4'][0], [3, 2, 0, 1]))

        self.conv1_g.bias.data = torch.from_numpy(np.squeeze(self.params_values_list['conv1'][1]))
        self.conv2_g.bias.data = torch.from_numpy(np.squeeze(self.params_values_list['conv2'][1]))
        self.conv3_g.bias.data = torch.from_numpy(np.squeeze(self.params_values_list['conv3'][1]))
        self.conv4_g.bias.data = torch.from_numpy(np.squeeze(self.params_values_list['conv4'][1]))

        self.fc1.weight.data = torch.from_numpy(np.transpose(self.params_values_list['fc1'][0], [1, 0]))
        self.fc1.bias.data = torch.from_numpy(np.squeeze(self.params_values_list['fc1'][1]))
        self.fc2.weight.data = torch.from_numpy(np.transpose(self.params_values_list['fc2'][0], [1, 0]))
        self.fc2.bias.data = torch.from_numpy(np.squeeze(self.params_values_list['fc2'][1]))


    def _import_model(self, net_path):
        return np.load(net_path, allow_pickle=True, encoding='latin1').item()


    def forward(self, xl, xg):
        xl = self.conv1_l(xl)
        xl = self.relu1_l(xl)
        xl = self.LRNl1(xl)
        xl = self.pool1_l(xl)
        xl = self.conv2_l(xl)
        xl = self.relu2_l(xl)
        xl = self.LRNl2(xl)
        xl = self.pool2_l(xl)
        xl = self.conv3_l(xl)
        xl = self.relu3_l(xl)
        xl = self.conv4_l(xl)
        xl = self.relu4_l(xl)

        xg = self.conv1_g(xg)
        xg = self.relu1_g(xg)
        xg = self.LRNg1(xg)
        xg = self.pool1_g(xg)
        xg = self.conv2_g(xg)
        xg = self.relu2_g(xg)
        xg = self.LRNg2(xg)
        xg = self.pool2_g(xg)
        xg = self.conv3_g(xg)
        xg = self.relu3_g(xg)
        xg = self.conv4_g(xg)
        xg = self.relu4_g(xg)

        x = torch.cat([xg, xl], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.out(x)

        return x

