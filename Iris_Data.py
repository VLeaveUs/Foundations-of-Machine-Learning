from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d
from numpy.core.multiarray import ndarray
from sklearn import datasets
import math
import scipy as sp
from sklearn.preprocessing import LabelEncoder

# Preprocessing
df = pd.read_csv('C:\\Users\\billl\\PycharmProjects\\Classification_&_LinearRegression_CW\\bezdekIris.data',
                 header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'Class'])
K = df.keys()
print(K)
X = df[['sepal length', 'sepal width', 'petal length', 'petal width']].values
y = df['Class'].values
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1
label_dict = {1: 'Iris-setosa', 2: 'Iris-versicolor', 3: 'Iris-virginica'}

class_mean = []
for i in range(1, 4):
    class_mean.append(np.mean(X[y == i], axis=0))
print('Class means are:\n', class_mean)

# Computing optimal projection direction
S_W = np.zeros((4, 4))
for c, cl_mean in zip(range(1, 4), class_mean):
    cov_i = np.zeros((4, 4))
    for j in (X[y == c]):
        j, cl_mean = j.reshape(4, 1), cl_mean.reshape(4, 1)
        cov_i += (j - cl_mean).dot((j - cl_mean).T)
    S_W += cov_i
print('Within Class Covariance Matrix:\n', S_W)

overall_mean = np.mean(X, axis=0)
S_B = np.zeros((4, 4))
for c, cl_mean in zip(range(1, 4), class_mean):
    N = len(X[y == c])
    overall_mean, cl_mean = overall_mean.reshape(4, 1), cl_mean.reshape(4, 1)
    S_B += N * (cl_mean - overall_mean).dot((cl_mean - overall_mean).T)
print('Between Class Covariance Matrix:\n', S_B)

Mat = np.linalg.inv(S_W).dot(S_B)
eig_v, eig_w = np.linalg.eig(Mat)
print('Eigenvalues are: ', eig_v, 'with corresponding Eigenvectors:\n', eig_w)
for v, i in zip(eig_v, range(4)):
    w = eig_w[:, i]
    w = w.reshape(4, 1)
    result = (np.linalg.inv(S_W).dot(S_B) - v*np.identity(4)).dot(w)
    print('Confirming result for eigenvector %i: ' % i, result)
for i in range(len(eig_v)):
    w = eig_w[:, i].reshape(4, 1)
    np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(w),
                                         eig_v[i] * np.identity(4).dot(w),
                                         decimal=6, err_msg='Wrong', verbose=True)
print('ok')

print('Variance explained:\n')
eigv_sum = 0
for v in eig_v:
    eigv_sum += np.abs(v)
for i, j in enumerate(eig_v):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (abs(j)/eigv_sum)))
W = np.hstack((eig_w[:, 0].reshape(4, 1), eig_w[:, 1].reshape(4, 1)))
print('Matrix W:\n', W)

Y = X.dot(W)

# Histograms of the three classes in the reduced dimensional space
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
for i, ax in enumerate(axes):
    for cl, clr in zip(range(1, 4), ('b', 'r', 'g')):
        ax.hist(Y[y == cl, i], 70, density=True, label='class %s' % label_dict[cl],
                color=clr, alpha=0.3)
        ax.legend(loc='upper right', fancybox=True, fontsize=8)
    ax.set_title('Feature %i' % i)
fig.suptitle('Iris LDA Distributions')
plt.show()
# for cl, clr in zip(range(1, 4), ('hot', 'cool', 'spring')):
#     plt.hist2d(Y[y == cl, 0], Y[y == cl, 1], 70, cmap=clr, )
#     plt.legend(labels='class %s' % label_dict[cl], loc='upper right', fancybox=True, fontsize=8)
# plt.show()