import  numpy as np
import pandas as pd
from sklearn import metrics
#read_data

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression,f_classif

#

class Analysis_of_variance(object):
    def __init__(self,x,y,k_th=20):
        self.x, self.y = x, y
        selector = SelectKBest(score_func=f_regression, k=k_th)
        print(x)
        print(y)
        results = selector.fit(x, y)

        self.results = results

        # 查看每个特征的分数和p-value
        # results.scores_: 每个特征的分数
        # results.pvalues_: 每个特征的p-value
        # results.get_support(): 返回一个布尔向量，True表示选择该特征，False表示放弃，也可以返回索引
        self.features = pd.DataFrame({
            "feature": x,
            "score": results.scores_,
            "pvalue": results.pvalues_,
            # "select": results.get_support()
        })
        self.features.sort_values("score", ascending=False)
        print(self.features.sort_values("score", ascending=False))
    def return_features(self):
        return self.features
    def return_index(self):
        x_new_index = self.results.get_support(indices=True)
        x_new = self.x.iloc[:, x_new_index]
        x_new.head()
        print(x_new)





if __name__=='__main__':
    Molecular = pd.read_excel('./data/Molecular_Descriptor.xlsx')
    ER = pd.read_excel('./data/ERα_activity.xlsx')
    data=Molecular.values
    data=np.array(data[:,1:],dtype = float)
    label = np.array(ER.values[:,2],dtype=float)
    function_1 = Analysis_of_variance(data,label)
    function_1.return_index()





