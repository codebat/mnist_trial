# -*- coding: utf-8 -*-
"""
Script for kaggle submission for mnist dataset

using scikit-learn implementation of k-Nearest neighbors 
"""

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

train_frame = pd.read_csv('train.csv')
feature = train_frame.iloc[:,1:]
label = train_frame.iloc[:,0]

pca = PCA(n_components=200)
modified_feature = pca.fit_transform(feature)

knn_classify = KNeighborsClassifier(n_neighbors=10,algorithm="auto")
knn_classify.fit(modified_feature,label)

test_frame = pd.read_csv('test.csv')
modified_test_feature = pca.fit_transform(test_frame)

knn_labels = knn_classify.predict(modified_test_feature)
knn_df = pd.DataFrame(knn_labels)
knn_df.index+=1
knn_df.to_csv('knn_10',header = ['Label'],index_label='ImageId')