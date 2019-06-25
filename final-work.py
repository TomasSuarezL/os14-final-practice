# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:17:01 2019

@author: Tomas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")

# Import Dataset from .csv with features and classifications 
dataset = pd.read_csv('CTGRawData.csv', sep=';')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 21].values

# Standarize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_ctg_scale = scaler.fit_transform(X)

# Perform PCA to find the most significant features
from sklearn.decomposition import PCA
# select the number of components
pca = PCA()
X_ctg_pca = pca.fit_transform(X_ctg_scale)  

sns.barplot(np.arange(np.shape(pca.explained_variance_ratio_)[0]),pca.explained_variance_ratio_)
plt.xlabel("Eigen values")
plt.ylabel("Explained variance")
plt.show()



