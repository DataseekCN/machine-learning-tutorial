# -*- coding: utf-8 -*-

"""
    版本:     1.0
    日期:     2018/11
    文件名:    utils.py
    功能：     工具文件
"""
import matplotlib.pyplot as plt
import seaborn as sns

from risk_rating import config
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


def inspect_dataset(train_data, test_data):
    """
        查看数据集

        参数：
            - train_data:   训练数据集
            - test_data:    测试数据集
    """
    print('\n===================== 数据查看 =====================')
    print('训练集有{}条记录。'.format(len(train_data)))
    print('测试集有{}条记录。'.format(len(test_data)))

    # 可视化各类别的数量统计图
    plt.figure(figsize=(10, 5))

    # 训练集
    ax1 = plt.subplot(1, 2, 1)
    sns.countplot(x=config.label_col, data=train_data)

    plt.title('Training set')
    plt.xlabel('Class')
    plt.ylabel('Count')

    # 测试集
    plt.subplot(1, 2, 2, sharey=ax1)
    sns.countplot(x=config.label_col, data=test_data)

    plt.title('Testing set')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()


def transform_train_data(train_data):
    """
        对训练数据进行特征工程处理
        1.  独热编码
        2.  范围归一化
        3.  特征降维
        参数：
            - train_data:   DataFrame训练数据

        返回：
            - X_train:  训练数据处理后的特征
            - label_encs:   标签 encoders
            - onehot_enc:   独热编码 encoders
            - scaler:       范围归一化 scaler
            - pca:          降维 pca
    """
    label_encs = []
    onehot_enc = OneHotEncoder(sparse=False)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 类别型数据
    label_feats = None
    for category_cols in config.category_cols:
        label_enc = LabelEncoder()
        label_feat = label_enc.fit_transform(train_data[category_cols].values).reshape(-1, 1)
        if label_feats is None:
            label_feats = label_feat
        else:
            label_feats = np.hstack((label_feats, label_feat))
        label_encs.append(label_enc)

    onehot_feats = onehot_enc.fit_transform(label_feats)

    # 数值型数据
    numeric_feats = train_data[config.num_cols].values

    # 合并所有特征
    all_feats = np.hstack((onehot_feats, numeric_feats))

    # 范围归一化
    scaled_all_feats = scaler.fit_transform(all_feats)

    print('特征处理后，特征维度为: {}（其中类别型特征维度为: {}，数值型特征维度为: {}）'.format(
        scaled_all_feats.shape[1], onehot_feats.shape[1], numeric_feats.shape[1]))

    # 使用特征降维
    pca = PCA(n_components=0.99)
    X_train = pca.fit_transform(scaled_all_feats)

    print('PCA特征降维后，特征维度为: {}'.format(X_train.shape[1]))

    return X_train, label_encs, onehot_enc, scaler, pca


def transform_test_data(test_data, label_encs, onehot_enc, scaler, pca):
    """

        参数：
            - test_data:   DataFrame训练数据
            - label_encs:   来自训练数据的标签 encoders
            - onehot_enc:   来自训练数据的独热编码 encoders
            - scaler:       来自训练数据的范围归一化 scaler
            - pca:          来自训练数据的降维 pca

        返回：
            - X_test:  转换后特征
    """
    # 类别型数据
    label_feats = None
    for i, category_cols in enumerate(config.category_cols):
        label_enc = label_encs[i]
        label_feat = label_enc.fit_transform(test_data[category_cols].values).reshape(-1, 1)
        if label_feats is None:
            label_feats = label_feat
        else:
            label_feats = np.hstack((label_feats, label_feat))

    onehot_feats = onehot_enc.transform(label_feats)

    # 数值型数据
    numeric_feats = test_data[config.num_cols].values

    # 合并所有特征
    all_feats = np.hstack((onehot_feats, numeric_feats))

    # 范围归一化
    scaled_all_feats = scaler.transform(all_feats)

    # 使用特征降维
    X_test = pca.transform(scaled_all_feats)

    return X_test


def transform_data(train_data, test_data):
    """
        将类别型特征通过独热编码进行转换
        将数值型特征范围归一化到0-1
        使用PCA进行特征降维

        参数：
            - train_data:   DataFrame训练数据
            - test_data:    DataFrame测试数据

        返回：
            - X_train:  训练数据处理后的特征
            - X_test:   测试数据处理后的特征
    """
    # 独热编码处理类别特征
    encoder = OneHotEncoder(sparse=False)
    X_train_cat_feat = encoder.fit_transform(train_data[config.category_cols].values)
    X_test_cat_feat = encoder.transform(test_data[config.category_cols].values)

    # 范围归一化处理数值型特征
    scaler = MinMaxScaler()
    X_train_num_feat = scaler.fit_transform(train_data[config.num_cols].values)
    X_test_num_feat = scaler.transform(test_data[config.num_cols].values)

    # 合并所有特征
    X_train_raw = np.hstack((X_train_cat_feat, X_train_num_feat))
    X_test_raw = np.hstack((X_test_cat_feat, X_test_num_feat))

    print('特征处理后，特征维度为: {}（其中类别型特征维度为: {}，数值型特征维度为: {}）'.format(
        X_train_raw.shape[1], X_train_cat_feat.shape[1], X_train_num_feat.shape[1]))

    # 使用特征降维
    pca = PCA(n_components=0.99)
    X_train = pca.fit_transform(X_train_raw)
    X_test = pca.transform(X_test_raw)

    print('PCA特征降维后，特征维度为: {}'.format(X_train.shape[1]))

    return X_train, X_test


def train_test_model(X_train, y_train, X_test, y_test, model_name, model, param_range):
    """
        训练并测试模型
        model_name:
            kNN         kNN模型，对应参数为 n_neighbors
            LR          逻辑回归模型，对应参数为 C
            SVM         支持向量机，对应参数为 C
            DT          决策树，对应参数为 max_depth
            Stacking    将kNN, SVM, DT集成的Stacking模型， meta分类器为LR
            AdaBoost    AdaBoost模型，对应参数为 n_estimators
            GBDT        GBDT模型，对应参数为 learning_rate
            RF          随机森林模型，对应参数为 n_estimators

        根据给定的参数训练模型，并返回
        1. 最优模型
        2. 平均训练耗时
        3. 准确率
    """
    print('训练{}...'.format(model_name))
    clf = GridSearchCV(estimator=model,
                       param_grid=param_range,
                       cv=5,
                       scoring='accuracy',
                       refit=True)
    start = time.time()
    clf.fit(X_train, y_train)
    # 计时
    end = time.time()
    duration = end - start
    print('耗时{:.4f}s'.format(duration))

    # 验证模型
    train_score = clf.score(X_train, y_train)
    print('训练准确率：{:.3f}%'.format(train_score * 100))

    test_score = clf.score(X_test, y_test)
    print('测试准确率：{:.3f}%'.format(test_score * 100))
    print('训练模型耗时: {:.4f}s'.format(duration))
    print()

    return clf, test_score, duration
