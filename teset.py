import collections
import math
from os import path

import pandas as pd

# #
Molecular_train = pd.read_excel('./data/Molecular_Descriptor.xlsx')
# # row,col = Molecular_train.shape
# #
# # print(row,col)
# # cnt = 0
# # dd = collections.defaultdict(int)
# # for i in range(1,col):
# #     for k in Molecular_train.iloc[:,i]:
# #         dd[(i,k)] += 1
# #         if dd[(i,k)] > row * 0.95:
# #             cnt += 1
# #             break
# # print(cnt)
#
#
Molecular_head = Molecular_train.columns[1:].values

result_dir = "result_4"
av_1_name = "av_1_ensemble.csv"
av_2_name = "av_2_ensemble.csv"
rc_1_name = "rc_1_ensemble.csv"
rc_2_name = "rc_2_ensemble.csv"
if_name = "if_ensemble.csv"
final_name = "final_ensemble.csv"

av_1 = pd.read_csv(path.join(result_dir,final_name))
av_1 = av_1["0"].values
for idx in av_1:
    print(Molecular_head[idx])

