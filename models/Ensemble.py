import torch

import torch.nn as nn
import torch.nn.functional as F

class Single_Linear(nn.Module):
    def __init__(self):
        super(Single_Linear, self).__init__()
        self.linear = torch.nn.Linear(20,1,bias=True)
    def forward(self,x):
        if isinstance(x,list):
            [x] = x
        y = self.linear(x)
        return y

class Multi_Linear(nn.Module):
    def __init__(self):
        super(Multi_Linear, self).__init__()
        self.linear = torch.nn.Linear(20, 10)
        self.tanh = torch.nn.Tanh()
        self.linear_2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        if isinstance(x,list):
            [x] = x
        y = self.linear(x)
        y = self.tanh(y)
        y = self.linear_2(y)

        return y



class Ensemble_model(nn.Module):
    def __init__(self):
        super(Ensemble_model, self).__init__()
        self.function_1 =  Single_Linear()
        self.function_2 = Multi_Linear()
        self.concat_linear = torch.nn.Linear(2,1)

    def forward(self, input):
        [x] = input
        # print(x.shape)
        y_1 = self.function_1(x)
        # print()
        y_2 = self.function_2(x)
        # print(y_1.shape)
        # print(y_2.shape)
        concat = torch.cat((y_1, y_2), dim=-1)
        y = self.concat_linear(concat)
        return  y
        


