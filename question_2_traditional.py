import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd
from sklearn.metrics import  mean_squared_error,r2_score

###########2.回归部分##########


###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree
import  codecs
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
ExtraTreeRegressor = ExtraTreeRegressor()

###########4.具体方法调用部分##########
if __name__ == '__main__':
    Molecular_train = pd.read_csv('./data/best_column.csv',header=0)
    Molecular_test = pd.read_csv('./data/test_best_column.csv',header=0)
    train_data_all: pd.DataFrame = Molecular_train.sample(frac=1.0)
    # train_data_all = Molecular_train
    rows, cols = train_data_all.shape

    split_index_1 = int(rows * 0.15)

    dev_data_split= train_data_all.iloc[:split_index_1, :]

    train_data_split= train_data_all.iloc[split_index_1: rows, :]
    x_train,y_train = train_data_split.iloc[:, 2:22],train_data_split.iloc[:,-1]
    x_dev, y_dev = dev_data_split.iloc[:, 2:22], dev_data_split.iloc[:, -1]


    model_list = []



    # model= model_DecisionTreeRegressor
    # model= ExtraTreeRegressor
    # model= model_LinearRegression
    model_1 = model_SVR
    # model = model_KNeighborsRegressor
    model_2 = model_RandomForestRegressor
    # model = model_AdaBoostRegressor
    model_3 = model_GradientBoostingRegressor
    model_4 = model_BaggingRegressor

    #
    model_1.fit(x_train,y_train)
    result_1 = model_1.predict(x_dev)

    model_2.fit(x_train, y_train)
    result_2 = model_2.predict(x_dev)

    model_3.fit(x_train, y_train)
    result_3 = model_3.predict(x_dev)

    model_4.fit(x_train, y_train)
    result_4 = model_4.predict(x_dev)
    result = (result_1+result_2+result_3+result_4)/4

    result_4 = model_4.predict(x_dev)
    result = (result_1+result_2+result_3+result_4)/4
    # model.fit(x_train, y_train)
    # result= model.predict(x_dev)

    print(mean_squared_error(y_dev,result))

    score = r2_score(y_dev,result)
    print(score)


    # save_path = "./result_2/question_2_traditional.text"
    # file_out = codecs.open(save_path, 'a+')
    # file_out.write(model.__class__.__name__)
    # file_out.write("\r")
    # file_out.write("determination R^2 of the Dev is {0}".format(score))
    # file_out.write("\r")
    # file_out.write(result_test)
    # file_out.close()
    # print(result)
    plt.figure()
    plt.plot(np.arange(len(result)), y_dev,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('score: %f'%score)
    plt.legend()
    plt.show()
    # plt.savefig("./figure/")