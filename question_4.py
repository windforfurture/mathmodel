from os import path

import joblib
import pandas as pd

from sklearn import tree, svm, ensemble
from sklearn.ensemble import BaggingRegressor, BaggingClassifier

# 3.1决策树分类
from sklearn.preprocessing import StandardScaler

model_DecisionTreeClassifier = tree.DecisionTreeClassifier()

# 3.3SVM回归
model_SVR = svm.SVR()

# 3.5随机森林回归
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)  # 这里使用20个决策树
model_RandomForestClassifier = ensemble.RandomForestClassifier(n_estimators=20)  # 这里使用20个决策树

# 3.6Adaboost回归
model_AdaBoostClassifier = ensemble.AdaBoostClassifier(n_estimators=50)  # 这里使用50个决策树

# 3.7GBRT回归
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树
model_GradientBoostingClassifier = ensemble.GradientBoostingClassifier(n_estimators=100)  # 这里使用100个决策树

# 3.8Bagging回归
model_BaggingRegressor = BaggingRegressor()
model_BaggingClassifier = BaggingClassifier()

r_ensemble_models = [model_SVR, model_RandomForestRegressor, model_GradientBoostingRegressor,
                             model_BaggingRegressor]
c_ensemble_models = [model_DecisionTreeClassifier, model_RandomForestClassifier, model_AdaBoostClassifier,
                              model_GradientBoostingClassifier, model_BaggingClassifier]


result_dir = 'result_4'
insect_name = "insect_ensemble.csv"
insect_df = pd.read_csv(path.join(result_dir, insect_name))
insect_l = insect_df["0"].values

Molecular_train = pd.read_excel('./data/Molecular_Descriptor.xlsx')
Molecular_head = Molecular_train.columns
ADMET_train = pd.read_excel('./data/ADMET.xlsx')
ADMET_train["Caco-2"] = ADMET_train["Caco-2"].astype(int)
ADMET_train["CYP3A4"] = ADMET_train["CYP3A4"].astype(int)
ADMET_train["hERG"] = ADMET_train["hERG"].astype(int)
ADMET_train["HOB"] = ADMET_train["HOB"].astype(int)
ADMET_train["MN"] = ADMET_train["MN"].astype(int)

original_data_all = Molecular_train.iloc[:,1:]


original_data_all = original_data_all.iloc[:, insect_l]

data_all = pd.DataFrame(StandardScaler().fit_transform(original_data_all))
data_all.insert(data_all.shape[1],"",Molecular_train.iloc[:,-1])
data_all = pd.concat([data_all,ADMET_train.iloc[:,1:]],axis=1)

train_data_all: pd.DataFrame = data_all.sample(frac=1.0)
rows, _ = train_data_all.shape

split_index_1 = int(rows * 0.2)

dev_data_split,train_data_split = train_data_all.iloc[:split_index_1, :], train_data_all.iloc[split_index_1:,:]

x_train,r_y_train,c_y_train = train_data_split.iloc[:, :-6],train_data_split.iloc[:,-6],train_data_split.iloc[:,-5:]
x_dev,r_y_dev,c_y_dev = dev_data_split.iloc[:, :-6],dev_data_split.iloc[:,-6],dev_data_split.iloc[:,-5:]


# # 获取回归的4个最佳模型
# r_best_ensemble_models = []
# for r_model in r_ensemble_models:
#     max_score = 0
#     best_model = None
#     for _ in range(5):
#         r_model.fit(r_x_train,r_y_train)
#         score = r_model.score(r_x_dev,r_y_dev)
#         if score > max_score:
#             best_model = r_model
#             max_score = score
#     r_best_ensemble_models.append(best_model)


# # 获取分类的5个最佳模型
# c_best_ensemble_models = []
# for c_model in c_ensemble_models:
#     c_c_best_ensemble_models = []
#     max_score = 0
#     best_model = None
#     for _ in range(5):
#         c_model.fit(c_x_train,c_y_train)
#         score = c_model.score(c_x_dev,c_y_dev)
#         if score > max_score:
#             best_model = r_model
#             max_score = score
#     c_c_best_ensemble_models.append(best_model)

# 取最佳的回归模型
r_best_model_file = path.join(result_dir, "r_model.pkl")
max_score = 0
for r_model in r_ensemble_models:
    for _ in range(5):
        r_model.fit(x_train,r_y_train)
        score = r_model.score(x_dev,r_y_dev)
        if score > max_score:
            max_score = score
            joblib.dump(r_model, r_best_model_file)

# 取最佳的5个分类模型
c_best_model_file_list = []

for i in range(5):
    c_best_model_file_list.append(path.join(result_dir,"c_model_"+str(i)+".pkl"))
    max_score = 0
    for c_model in c_ensemble_models:
        for _ in range(5):
            c_model.fit(x_train,c_y_train.iloc[:,i])
            score = c_model.score(x_dev,c_y_dev.iloc[:,i])
            if score > max_score:
                max_score = score
                joblib.dump(c_model, c_best_model_file_list[i])

# 获取列的取值范围
max_l = []
min_l = []
good_label = [1, 1, 0, 1, 0]
n_f = len(insect_l)

for i_data in data_all.values:
    good_sum = 0
    for i in range(5):
        if good_label[i] == i_data[-5+i]:
            good_sum += 1
    if good_sum >= 3:

        if len(max_l) == 0:

            max_l += list(i_data[:-6])
            min_l += list(i_data[:-6])
        else:
            for j in range(n_f):
                if i_data[j] > max_l[j]:
                    max_l[j] = i_data[j]
                if i_data[j] < min_l[j]:
                    min_l[j] = i_data[j]

# 将取值切分

sep_ratio = 0.1
sep_num = int(1/sep_ratio) + 1

all_data = []
for i in range(n_f):
    sep_value = (max_l[i] - min_l[i]) * sep_ratio
    a_feature_data = []
    a_first_value = min_l[i]
    for j in range(sep_num):
        a_feature_data.append(a_first_value)
        a_first_value += sep_value
    all_data.append(a_feature_data)

max_yz = 0
best_test = None
k = 0
c_best_model_list = []
for i in range(5):
    c_best_model_list.append(joblib.load(c_best_model_file_list[i]))
r_best_model = joblib.load(r_best_model_file)

for i_1 in range(n_f):
    for i_2 in range(sep_num):
        for i_3 in range(sep_num):
            for i_4 in range(sep_num):
                for i_5 in range(sep_num):
                    x_test = [all_data[0][i_1],all_data[1][i_2],all_data[2][i_3],all_data[3][i_4],all_data[3][i_5]]
                    c_y_test_sum = 0
                    for i in range(5):
                        c_best_model = c_best_model_list[i]
                        c_y_test_predict = c_best_model.predict([x_test])[0]
                        if c_y_test_predict == good_label[i]:
                            c_y_test_sum += 1
                    if c_y_test_sum >= 3:
                        yz = r_best_model.predict([x_test])[0]
                        if yz > max_yz:
                            max_yz = yz
                            best_test = x_test

# 反归一化
best_original = []
means = original_data_all.mean().values
stds = original_data_all.std().values

Molecular_head = Molecular_head[1:]
columns_name = []
for i_sect in insect_l:
    columns_name.append(Molecular_head[i_sect])
for i in range(n_f):
    best_original.append([columns_name[i],best_test[i] * stds[i] + means[i]])
print(best_original)




















