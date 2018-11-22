# -*- coding: utf-8 -*-

"""
    版本:     1.0
    日期:     2018/11
    文件名:    main.py
    功能：     主程序
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from mlxtend.classifier import StackingClassifier

from risk_rating import config
from risk_rating import utils


def main():
    """
        主函数
    """
    # 加载数据
    raw_data = pd.read_csv(os.path.join(config.dataset_path, 'zoo.csv'), usecols=config.all_cols)

    # 分割数据集
    train_data, test_data = train_test_split(raw_data, test_size=1/4, random_state=10)

    # 标签
    y_train = train_data[config.label_col].values
    y_test = test_data[config.label_col].values

    # 数据查看
    utils.inspect_dataset(train_data, test_data)

    # 特征工程
    print('\n===================== 特征工程 =====================')
    X_train, label_encs, onehot_enc, scaler, pca = utils.transform_train_data(train_data)
    X_test = utils.transform_test_data(test_data, label_encs, onehot_enc, scaler, pca)

    # 构建训练测试数据
    # 数据建模及验证
    print('\n===================== 数据建模及验证 =====================')

    sclf = StackingClassifier(classifiers=[KNeighborsClassifier(),
                                           SVC(),
                                           DecisionTreeClassifier()],
                              meta_classifier=LogisticRegression())

    model_name_param_dict = {'kNN': (KNeighborsClassifier(),
                                     {'n_neighbors': [5, 25, 55]}),
                             'LR': (LogisticRegression(),
                                    {'C': [0.01, 1, 100]}),
                             'SVM': (SVC(),
                                     {'C': [0.01, 1, 100]}),
                             'DT': (DecisionTreeClassifier(),
                                    {'max_depth': [50, 100, 150]}),
                             'Stacking': (sclf,
                                          {'kneighborsclassifier__n_neighbors': [5, 25, 55],
                                           'svc__C': [0.01, 1, 100],
                                           'decisiontreeclassifier__max_depth': [50, 100, 150],
                                           'meta-logisticregression__C': [0.01, 1, 100]}),
                             'AdaBoost': (AdaBoostClassifier(),
                                          {'n_estimators': [50, 100, 150, 200]}),
                             'GBDT': (GradientBoostingClassifier(),
                                      {'learning_rate': [0.01, 0.1, 1, 10, 100]}),
                             'RF': (RandomForestClassifier(),
                                    {'n_estimators': [100, 150, 200, 250]})
                             }

    # 比较结果的DataFrame
    results_df = pd.DataFrame(columns=['Accuracy (%)', 'Time (s)'],
                              index=list(model_name_param_dict.keys()))
    results_df.index.name = 'Model'
    for model_name, (model, param_range) in model_name_param_dict.items():
        _, best_acc, mean_duration = utils.train_test_model(X_train, y_train, X_test, y_test,
                                                            model_name, model, param_range)
        results_df.loc[model_name, 'Accuracy (%)'] = best_acc * 100
        results_df.loc[model_name, 'Time (s)'] = mean_duration

    results_df.to_csv(os.path.join(config.output_path, 'model_comparison.csv'))

    # 模型及结果比较
    print('\n===================== 模型及结果比较 =====================')

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    results_df.plot(y=['Accuracy (%)'], kind='bar', ylim=[0, 1], ax=ax1, title='Accuracy(%)', legend=False)

    ax2 = plt.subplot(1, 2, 2)
    results_df.plot(y=['Time (s)'], kind='bar', ax=ax2, title='Time(s)', legend=False)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_path, 'pred_results.png'))
    plt.show()


if __name__ == '__main__':
    main()
