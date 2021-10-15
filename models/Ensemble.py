import torch

import torch.nn as nn
import torch.nn.functional as F

class Singal_Liner(nn.Module):
    def __init__(self):
        super(Singal_Liner, self).__init__()
        self.linear = torch.nn.Linear(20,1)
    def forward(self,x):
        y = self.linear(x)
        return y

class Multi_liner(nn.Module):
    def __init__(self):
        super(Multi_liner, self).__init__()
        self.linear = torch.nn.Linear(20, 50)
        self.tanh = torch.nn.Tanh()
        self.linear_2 = torch.nn.Linear(50, 1)

    def forward(self, x):
        y = self.linear(x)
        y = self.tanh(y)
        y = self.linear_2(y)

        return y



class Ensemble_model(nn.Module):
    def __init__(self):
        super(Ensemble_model, self).__init__()
        self.function_1 =  Singal_Liner()
        self.function_2 = Multi_liner()
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
        


