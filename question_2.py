import torch
import  numpy as np
import  pandas as pd
import argparse
import  parser
import codecs
from  data_uitls import  DatesetReader
from bucket_iterator import  BucketIterator
class Instructor:
    def __init__(self, opt,train_data):
        char_Tokenizer = DatesetReader(train_data)

        train_data: pd.DataFrame = train_data.sample(frac=1.0)
        rows, cols = train_data.shape
        split_index_1 = int(rows * 0.2)

        dev_data_split= train_data.iloc[:split_index_1, :]
        train_data_split= train_data.iloc[split_index_1: rows, :]
        train_data =char_Tokenizer.__read_data__(train_data_split)
        dev_data = char_Tokenizer.__read_data__(dev_data_split)



        self.train_data_loader = BucketIterator(data=train_data, batch_size=opt.batch_size, shuffle=True,
                                                sort=True)

        self.dev_data_loader = BucketIterator(data=dev_data, batch_size=opt.batch_size, shuffle=True,
                                                sort=True)
        self._print_args()
        self.global_

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











if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_char_embedding', default=True,type=bool)
    parser.add_argument('--char_model',default="BILST",type=str)
    parser.add_argument('--model_name', default="dense",type=str)
    parser.add_argument('--Pattern', default="train", type=str) #"train","eval","test"
    parser.add_argument('--batch_size', default=32, type=int)

    opt = parser.parse_args()


    Molecular_train = pd.read_excel('./data/Molecular_Descriptor.xlsx')
    ER_train = pd.read_excel('./data/ERα_activity.xlsx')

    Molecular_test = pd.read_excel('./data/Molecular_Descriptor.xlsx',sheet_name="test")
    ER_test = pd.read_excel('./data/ERα_activity.xlsx',sheet_name="test")

    Molecular_train.head()


    train_data = Molecular_train


    test_data = Molecular_test



    ins = Instructor(opt,train_data)
    if opt.Pattern =="train":
        best= ins.trian()

    f_out = codecs.open('log/' + opt.model_name+ '_' + opt.use_char_embedding+ '_val.txt',
                        'a+', encoding="utf-8")

    f_out.write('max_test_acc_avg: {0}, max_test_f1_avg: {1}\n'.format(max_test_acc_avg / repeats,
                                                                       max_test_f1_avg / repeats))
    f_out.write("\n")

