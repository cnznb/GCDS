# -*- encoding = utf-8 -*-
"""
@description: 用DT训练并对REMS数据进行测试
@date: 2022/9/26
@File : DT.py
@Software : PyCharm
"""
import csv
import os
import sys
import numpy as np
import torch
from sklearn import tree
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
import datetime
import utils
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
data_path = 'E:/datasets'  # 训练数据集文件路径

# codeBERT   codeT5   codeGPT  graphCodeBERT   PLBART   coTexT
emb = 'codeBERT'
X_train = []
y_train = []
X_test = []
y_test = []


# def load_datas():
#     global X_train
#     global y_test
#     global X_test
#     global y_train
#     with open("C:/Users/rain_n/Desktop/train.csv", 'r') as sc:
#         reader = csv.reader(sc)
#         for row in reader:
#             vector = [float(x) for x in row[:128]]
#             label = int(row[128])
#             X_train.append(vector)
#             y_train.append(label)
#     with open("C:/Users/rain_n/Desktop/test.csv", 'r') as sc:
#         reader = csv.reader(sc)
#         for row in reader:
#             vector = [float(x) for x in row[:128]]
#             label = int(row[128])
#             X_test.append(vector)
#             y_test.append(label)
#     X_train = pd.DataFrame(X_train, columns=None)
#     y_train = pd.DataFrame(y_train, columns=None)
#     X_test = pd.DataFrame(X_test, columns=None)
#     y_test = pd.DataFrame(y_test, columns=None)
#     #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
#     #定义SMOTE模型，random_state相当于随机数种子的作用
#     smo = SMOTE(random_state=42)
#     X_train, y_train = smo.fit_resample(X_train, y_train)
#     return X_train, y_train, X_test, y_test


def load_data(src):
    """
    加载训练数据集并划分训练集和测试集
    :param src: 训练集文件
    :return: 训练集的vec和label
    """
    global X_train
    global y_test
    global X_test
    global y_train
    def train_search(path):
        global X_train
        global y_train
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if file_name == '[test]ganttproject' or file_name == '[test]xerces':
                continue
            if os.path.isdir(file_path):
                if file_name.startswith('1'):
                    print(file_path)
                    vec, lab = utils.solve(file_path, emb)
                    X_train += vec
                    y_train += lab
                else:
                    train_search(file_path)
    train_search(src)
    def test_search(path):
        global X_test
        global y_test
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isdir(file_path):
                if file_name.startswith('1'):
                    print(file_path)
                    vec, lab = utils.solve(file_path, emb)
                    X_test += vec
                    y_test += lab
                    #print(X[0])
                    # print(X)
                    # print(y)
                else:
                    test_search(file_path)
    test_search('E:/datasets/[test]xerces')
    test_search('E:/datasets/[test]ganttproject')
    X_train = pd.DataFrame(X_train, columns=None)
    y_train = pd.DataFrame(y_train, columns=None)
    X_test = pd.DataFrame(X_test, columns=None)
    y_test = pd.DataFrame(y_test, columns=None)
    # 定义SMOTE模型，random_state相当于随机数种子的作用
    smo = SMOTE(random_state=42)
    X_train, y_train = smo.fit_resample(X_train, y_train)
    return X_train, y_train, X_test, y_test


def test(estimator, X_test, y_test, nm):
    """
    分别测试测试目录下的所有项目向量文件并写入对应的结果文件中
    :param estimator: 训练得到的最优模型
    :param filepath: 测试集文件路径
    :param trainingfile: 训练模型所使用的训练数据集，这里仅用于对输出结果文件命名
    :return: None
    """
    # 测试数据

    y_pre = estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pre)
    precision = precision_score(y_test, y_pre, labels=None, pos_label=1, average='binary', sample_weight=None)
    recall = recall_score(y_test, y_pre, labels=None, pos_label=1, average='binary', sample_weight=None)
    f1 = f1_score(y_test, y_pre, labels=None, pos_label=1, average='binary', sample_weight=None)
    output = open("e_" + emb + ".txt", "a+")
    output.write(
        "-----------------------------------------------------------------------------------------------\n")
    output.write("Testing type : " + nm + "\n")
    output.write("Testing time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    output.write("best_estimator_ : " + str(estimator.best_estimator_) + "\n")
    output.write("accuracy: {:.3}, precision: {:.3}, recall: {:.3}, f1:{:.3}".format(accuracy, precision, recall, f1) + "\n")
    output.write(
        "-----------------------------------------------------------------------------------------------\n\n\n")
    output.close()


def DT_train(X_train, y_train, X_test, y_test):
    """
    运用网格搜索寻找最优参数，再对最优模型进行测试
    :param X_train:
    :param y_train:
    :return: None
    """
    # 网格搜索参数列表
    tuned_parameters = {
            'splitter': ('best', 'random'), # , 'random'
            'criterion': ("gini", "entropy"),
            "max_depth": [*range(30, 51, 10)],
            'min_samples_leaf': [*range(1, 10, 1)]
    }
    # 生成模型
    print("Start trainging : " + "\n")
    #grid = tree.DecisionTreeClassifier()  #GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='roc_auc', verbose=2, n_jobs=4)
    grid = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='roc_auc', verbose=2, n_jobs=4)
    # 把数据交给模型训练
    grid.fit(X_train, y_train)
    #print(grid.best_estimator_)
    test(grid, X_test, y_test, 'DT')


def SVM_train(X_train, y_train, X_test, y_test):
    """
    运用网格搜索寻找最优参数，再对最优模型进行测试
    :param X_train:
    :param y_train:
    :return: None
    """
    # 网格搜索参数列表
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3], 'C': [100]},
                        {'kernel': ['linear'], 'C': [1]}]
    # 生成模型
    print("Start trainging : " + "\n")
    y_train = np.array(y_train).ravel()
    y_train = np.reshape(y_train, (len(y_train),))
    grid = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='roc_auc', verbose=2, n_jobs=4)
    # 把数据交给模型训练
    grid.fit(X_train, y_train)
    test(grid, X_test, y_test, 'SVM')


def XGB_train(X_train, y_train, X_test, y_test):
    """
    运用网格搜索寻找最优参数，再对最优模型进行测试
    :param X_train:
    :param y_train:
    :return: None
    """
    # 网格搜索参数列表
    tuned_parameters = {
            'n_estimators': range(300, 401, 100),
            'max_depth': range(4, 15, 2),
            'learning_rate': np.linspace(0.1, 2, 8),
            'subsample': np.linspace(0.7, 0.9, 4),
            'colsample_bytree': np.linspace(0.8, 1, 5),
            'min_child_weight': range(1, 11, 2)
    }
    # 生成模型
    print("Start trainging : " + "\n")
    grid = GridSearchCV(XGBClassifier(eval_metric=['logloss', 'auc', 'error']), tuned_parameters, cv=5, scoring='roc_auc', verbose=2, n_jobs=4)
    # 把数据交给模型训练
    grid.fit(X_train, y_train)
    test(grid, X_test, y_test, 'XGB')

def RF_train(X_train, y_train, X_test, y_test):
    """
    运用网格搜索寻找最优参数，再对最优模型进行测试
    :param X_train:
    :param y_train:
    :return: None
    """
    # 网格搜索参数列表
    tuned_parameters = [
        {'n_estimators': [80, 90, 100, 50, 60, 70, 75], 'max_features': [8, 12, 10, 15, 17, 5, 4, 3, 2]},
        {'bootstrap': [False], 'n_estimators': [3, 7, 10], 'max_features': [2, 3, 4]},
    ]
    # 生成模型
    print("Start trainging : " + "\n")
    y_train = np.array(y_train).ravel()
    y_train = np.reshape(y_train, (len(y_train),))
    grid = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='roc_auc', verbose=2, n_jobs=4)
    # 把数据交给模型训练
    grid.fit(X_train, y_train)
    test(grid, X_test, y_test, 'RF')

def KNN_train(X_train, y_train, X_test, y_test):
    """
    运用网格搜索寻找最优参数，再对最优模型进行测试
    :param X_train:
    :param y_train:
    :return: None
    """
    # 网格搜索参数列表
    tuned_parameters = {
            "n_neighbors": range(1, 20),
            "weights": ['uniform', 'distance']
    }
    # 生成模型
    print("Start trainging : " + "\n")
    y_train = np.array(y_train).ravel()
    y_train = np.reshape(y_train, (len(y_train),))
    grid = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='roc_auc', verbose=2, n_jobs=4)
    # 把数据交给模型训练
    grid.fit(X_train, y_train)
    test(grid, X_test, y_test, 'KNN')


def LR_train(X_train, y_train, X_test, y_test):
    """
    运用网格搜索寻找最优参数，再对最优模型进行测试
    :param X_train:
    :param y_train:
    :return: None
    """
    # 网格搜索参数列表
    tuned_parameters = {
            'C': np.append(np.arange(0.01, 0.1, 0.01), [0.105]),
            'max_iter': range(120, 201, 20),
            'tol': [0.0001, 0.001, 0.01, 0.1]
    }
    # 生成模型
    print("Start trainging : " + "\n")
    y_train = np.array(y_train).ravel()
    y_train = np.reshape(y_train, (len(y_train),))
    grid = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5, scoring='roc_auc', verbose=2, n_jobs=4)
    # 把数据交给模型训练
    grid.fit(X_train, y_train)
    test(grid, X_test, y_test, 'LR')

def NB_train(X_train, y_train, X_test, y_test):
    """
    运用网格搜索寻找最优参数，再对最优模型进行测试
    :param X_train:
    :param y_train:
    :return: None
    """
    # 网格搜索参数列表
    tuned_parameters = {
        "alpha": np.arange(0.1, 2, 0.1)
    }
    # 生成模型
    print("Start trainging : " + "\n")
    y_train = np.array(y_train).ravel()
    y_train = np.reshape(y_train, (len(y_train),))
    grid = GridSearchCV(BernoulliNB(), tuned_parameters, cv=5, scoring='roc_auc', verbose=2, n_jobs=4)
    # 把数据交给模型训练
    grid.fit(X_train, y_train)
    test(grid, X_test, y_test, 'NB')


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    DT_train(X_train, y_train, X_test, y_test)
    KNN_train(X_train, y_train, X_test, y_test)
    LR_train(X_train, y_train, X_test, y_test)
    NB_train(X_train, y_train, X_test, y_test)
    RF_train(X_train, y_train, X_test, y_test)
    SVM_train(X_train, y_train, X_test, y_test)
    XGB_train(X_train, y_train, X_test, y_test)