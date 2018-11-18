# -*- coding: utf-8 -*-

"""
    版本:     1.0
    日期:     2018/11
    文件名:    config.py
    功能：     配置文件
"""
import os

# 指定数据集路径
dataset_path = './data'

# 结果保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 类别型特征(未补充完整)
category_cols = ['nation']

# 数值型特征(未补充完整)
num_cols = ['absent']

# 标签列
label_col = 'class_type'

# 需要读取的列
all_cols = category_cols + num_cols + [label_col]

