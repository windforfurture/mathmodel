import torch
import  torch.nn.functional as F
import  torch.nn as nn
import  numpy as np
import  pandas as pd
import argparse
import  parser
import codecs
from  data_uitls import  DatesetReader
from bucket_iterator import  BucketIterator
from models.Ensemble import  Ensemble_model,Single_Linear,Multi_linear
from sklearn.preprocessing import StandardScaler

class Instructor:
    def __init__(self, opt,train_data):
        self.opt = opt
        char_Tokenizer = DatesetReader(train_data)

        train_data: pd.DataFrame = train_data.sample(frac=1.0)
        rows, cols = train_data.shape
        split_index_1 = int(rows * 0.2)

        dev_data_split= train_data.iloc[:split_index_1, :]
        train_data_split= train_data.iloc[split_index_1: rows, :]
        # print(dev_data_split)
        # print(train_data_split)
        # exit()
        train_data =char_Tokenizer.__read_data__(train_data_split)
        dev_data = char_Tokenizer.__read_data__(dev_data_split)


        self.train_data_loader = BucketIterator(data=train_data, batch_size=opt.batch_size, shuffle=True,
                                                sort=True)

        self.dev_data_loader = BucketIterator(data=dev_data, batch_size=opt.batch_size, shuffle=True,
                                                sort=True)
        self.criterion = nn.MSELoss()
        if  self.opt.model_name ==  "Ensemble_model":
            self.model = Ensemble_model().to(device=opt.device)
        if self.opt.model_name =="Singal_Linear":
            self.model = Single_Linear().to(device=opt.device)
        if self.opt.model_name =="Multi_linear":
            self.model = Multi_Linear().to(device=opt.device)

        self._print_args()
        print("1111111111")
        self.global_mes_error=0.0

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=0))

    def _print_args(self):
        # print(self.model.gat_layer_list)
        # print(self.model.ModuleList)
        # exit()
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _eval(self):
        self.model.eval()
        dev_total_loss=0.0
        with torch.no_grad():
            for dev_i_batch, dev_sample_batched in enumerate(self.dev_data_loader):
                dev_inputs = [dev_sample_batched["batch_other_feature"].to(self.opt.device)]
                dev_targets = dev_sample_batched['batch_tags'].to(self.opt.device)
                outputs = self.model(dev_inputs)
                loss = self.criterion(outputs, dev_targets)
                dev_total_loss += loss
        print('\r >>> this epoch dev loss is {:.4f}'.format(dev_total_loss/(dev_i_batch*len(dev_sample_batched))))
        return  dev_total_loss

    def _train(self):
        self.model.train()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(_params, lr=0.00005)

        best_min_loss = 0.0
        for epoch in range(self.opt.num_epoch):
            train_total_loss = 0.0
            self.model.train()
            for train_i_batch, sample_batched in enumerate(self.train_data_loader):
                # print(sample_batched)
                # exit()
                inputs = [sample_batched["batch_other_feature"].to(self.opt.device)]
                targets = sample_batched['batch_tags'].to(self.opt.device)
                outputs= self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_total_loss+=loss
            print('\r >>> this epoch {0} train loss is {1}'.format(epoch,train_total_loss/(train_i_batch*len(sample_batched))))
            dev_total_loss = self._eval()
            if best_min_loss<dev_total_loss:
                torch.save(self.model,
                          'state_dict/' + str(opt.model_name)+ str(
                              best_min_loss) + '.pkl')
















if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_char_embedding', default=True,type=bool)
    parser.add_argument('--char_model',default="BILST",type=str)
    parser.add_argument('--Pattern', default="train", type=str) #"train","eval","test"
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--num_epoch',default=100,type=int)
    parser.add_argument('--model_name', default="Ensemble_model", type=str)
    opt = parser.parse_args()

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    Molecular_train = pd.read_excel('./data/Molecular_Descriptor.xlsx')
    ER_train = pd.read_excel('./data/ERα_activity.xlsx')

    Molecular_test = pd.read_excel('./data/Molecular_Descriptor.xlsx',sheet_name="test")
    ER_test = pd.read_excel('./data/ERα_activity.xlsx',sheet_name="test")

    Molecular_train.head()


    train_data = Molecular_train
    train_data = pd.DataFrame(StandardScaler().fit_transform(train_data))



    test_data = Molecular_test




    ins = Instructor(opt,train_data)
    print("this train")
    if opt.Pattern =="train":
        best_loss= ins._train()


