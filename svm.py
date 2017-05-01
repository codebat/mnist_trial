# -*- coding: utf-8 -*-
"""
Script for kaggle submission for mnist dataset

Using linear svm classifier of scikit-learn and principal component analysis

"""
from sklearn.decomposition import PCA

from sklearn import svm

import pandas as pd

train_frame = pd.read_csv('train.csv')
feature = train_frame.iloc[:,1:]
label = train_frame.iloc[:,0]

pca = PCA(n_components=200)
modified_feature = pca.fit_transform(feature)
classify = svm.LinearSVC()
classify.fit(modified_feature,label)

test_frame = pd.read_csv('test.csv')
modified_test_feature = pca.fit_transform(test_frame)

resulted_label = classify.predict(modified_test_feature)
result_df = pd.DataFrame(resulted_label)

result_df.index+=1
result_df.to_csv('linsvm_pca',header = ['Label'],index_label = 'ImageId')

# to be modified to take the components in PCA as command line argument