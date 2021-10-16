from os import path

import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR, SVC


class SelectFeatures(object):
    def __init__(self, X, y, k_th=30):
        self.X, self.y, self.k_th = X, y, k_th
        self.features = None
        self.results = None

    def analysis_of_variance(self):
        selector = SelectKBest(score_func=f_classif, k=self.k_th)
        results = selector.fit(self.X, self.y)
        self.results = results

        # 查看每个特征的分数和p-value
        # results.scores_: 每个特征的分数
        # results.pvalues_: 每个特征的p-value
        # results.get_support(): 返回一个布尔向量，True表示选择该特征，False表示放弃，也可以返回索引
        features = pd.DataFrame({
            "feature": self.X.columns,
            "score": results.scores_,
            "pvalue": results.pvalues_,
            "select": results.get_support()
        })
        features = features.sort_values(by=["score"], ascending=False)
        results = list(features.index)
        self.results = results[:self.k_th]

    def recursion_clear(self, p_estimator):
        # 创建筛选器
        selector = RFE(
            # estimator=LinearRegression()
            estimator=p_estimator,  # 由于这是分类问题，选择简单的逻辑回归
            n_features_to_select=self.k_th,  # 选择的最小特征数量
            # cv=5,  # 交叉验证折数
            # scoring="accuracy",  # 评估预测精度的指标
            # n_jobs=-1  # 使用所有CPU运算
        )

        # 拟合数据
        results = selector.fit(self.X, self.y)

        # 查看结果
        # results.n_features_: 最终选择的特征数量
        # results.support_: 布尔向量，True表示保留特征，False表示剔除特征
        # results.ranking_: 特征等级，1表示最高优先级，即应该保留的特征
        # print("Number of selected features = %d" % results.n_features_)
        # print("Selected features: %s" % results.support_)
        # print("Feature ranking: %s" % results.ranking_)
        nr = len(results.support_)
        results_list = []
        for i in range(nr):
            if results.support_[i]:
                results_list.append(i)
        self.results = results_list

    def importance_of_features(self):
        # 创建分类器
        model = RandomForestClassifier(random_state=123)

        # 拟合数据
        model.fit(self.X, self.y)

        # 提取特征重要性
        importance = pd.Series(model.feature_importances_)
        importance = importance.sort_values()
        results = pd.DataFrame(importance)
        # results = importance.iloc[-1]
        results = list(results.index)
        results.reverse()
        self.results = results[:self.k_th]

    def get_results(self):
        return self.results


def get_name_idx(result_list, head_list):
    ret = []
    for i_idx in result_list:
        ret.append([i_idx[0], head_list[i_idx[0]]]+i_idx[1:])
    return ret



def write_csv(p_result,p_file_name):
    df = pd.DataFrame(p_result)
    df.to_csv(p_file_name)


if __name__ == '__main__':
    Molecular = pd.read_excel('./data/Molecular_Descriptor.xlsx')
    ADMET = pd.read_excel('./data/ADMET.xlsx')
    # ER = pd.read_excel('./data/ERα_activity.xlsx')
    Molecular.head()
    data = Molecular.iloc[:, 1:-1]
    # 归一化处理
    my_head = data.columns.values.tolist()
    label_list = []
    for i in range(5):
        label_list.append(ADMET.iloc[:, 1+i])
    good_label = [1, 1, 0, 1, 0]
    nl = len(label_list[0])
    ensemble_label = [0] * nl
    for i in range(5):
        for j in range(nl):
            if label_list[i][j] == good_label[i]:
                ensemble_label[j] += 1
    for i in range(nl):
        if ensemble_label[i] >= 3:
            ensemble_label[i] = 1
        else:
            ensemble_label[i] = 0

    data = pd.DataFrame(StandardScaler().fit_transform(data))
    data.columns = my_head
    # 前30
    top_k_nums = 30


    # 文件设置
    result_dir = "result_4"
    regression_final_result_name = 'result_1/final.csv'

    """
        集成分类结果，>=3为1，否则为0
    """
    av_name = "av_ensemble.csv"
    rc_1_name = "rc_1_ensemble.csv"
    rc_2_name = "rc_2_ensemble.csv"
    if_name = "if_ensemble.csv"
    final_name = "final_ensemble.csv"
    insect_name = "insect_ensemble.csv"
    # function_1 = SelectFeatures(data, ensemble_label, k_th=top_k_nums)
    # # 集成，av与if有排名，rc无排名
    # ensemble_dict = dict()
    #
    # function_1.analysis_of_variance()
    # av_result = function_1.get_results()
    # ensemble(av_result)
    # write_csv(av_result, path.join(result_dir, av_name))
    #
    # estimator = SVC(kernel="linear")
    # function_1.recursion_clear(estimator)
    # rc_1_result = function_1.get_results()
    # ensemble(rc_1_result, is_score=False)
    # write_csv(rc_1_result, path.join(result_dir, rc_1_name))
    #
    # estimator = LogisticRegression()
    # function_1.recursion_clear(estimator)
    # rc_2_result = function_1.get_results()
    # ensemble(rc_2_result, is_score=False)
    # write_csv(rc_2_result, path.join(result_dir, rc_2_name))
    #
    # function_1.importance_of_features()
    # if_result = function_1.get_results()
    # ensemble(if_result)
    # write_csv(if_result, path.join(result_dir, if_name))
    #
    # final_result = []
    #
    # for key, value in ensemble_dict.items():
    #     final_result.append([key] + value)
    #
    # final_result.sort(key=lambda x: [-x[1], x[2]])
    # final_result = get_name_idx(final_result, my_head)
    # write_csv(final_result, path.join(result_dir, final_name))

    classifier_final_result_name = path.join(result_dir, final_name)
    cdf = pd.read_csv(classifier_final_result_name)
    rdf = pd.read_csv(regression_final_result_name)

    cdf_l = cdf.iloc[:20,1]
    rdf_s = set(rdf.iloc[:,1])
    print(cdf_l)
    print(rdf_s)
    insect_l = []
    for i_cdf in cdf_l:
        if i_cdf in rdf_s:
            insect_l.append(i_cdf)
    write_csv(insect_l, path.join(result_dir, insect_name))


    """
        5个种类分别计算相关
    """
    # insect_name = "insect_1_5.csv"
    # rdf = pd.read_csv(regression_final_result_name)
    # rdf_s = set(rdf.iloc[:20, 1])
    # for i in range(5):
    #     av_name = str(i)+"_" + "av.csv"
    #     rc_1_name = str(i) + "_" + "rc_1.csv"
    #     rc_2_name = str(i) + "_" + "rc_2.csv"
    #     if_name = str(i) + "_" + "if.csv"
    #     final_name = str(i) + "_" + "final.csv"
    #
    #     function_1 = SelectFeatures(data, label_list[i], k_th=top_k_nums)
    #     # 集成，av与if有排名，rc无排名
    #     ensemble_dict = dict()
    #
    #     function_1.analysis_of_variance()
    #     av_result = function_1.get_results()
    #     ensemble(av_result)
    #     write_csv(av_result, path.join(result_dir, av_name))
    #
    #     estimator = SVC(kernel="linear")
    #     function_1.recursion_clear(estimator)
    #     rc_1_result = function_1.get_results()
    #     ensemble(rc_1_result, is_score=False)
    #     write_csv(rc_1_result, path.join(result_dir, rc_1_name))
    #
    #     estimator = LogisticRegression()
    #     function_1.recursion_clear(estimator)
    #     rc_2_result = function_1.get_results()
    #     ensemble(rc_2_result, is_score=False)
    #     write_csv(rc_2_result, path.join(result_dir, rc_2_name))
    #
    #     function_1.importance_of_features()
    #     if_result = function_1.get_results()
    #     ensemble(if_result)
    #     write_csv(if_result, path.join(result_dir, if_name))
    #
    #     final_result = []
    #
    #     for key, value in ensemble_dict.items():
    #         final_result.append([key]+value)
    #
    #     final_result.sort(key=lambda x: [-x[1], x[2]])
    #     final_result = get_name_idx(final_result,my_head)
    #     write_csv(final_result, path.join(result_dir, final_name))





