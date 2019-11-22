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

x = np.linspace(0, 10, 200)
mean_a = np.array([6, 0])
mean_b = np.array([-6, 0])
covA = np.array([[5, 0], [0, 5]])
covB = np.array([[5, 0], [0, 5]])

N = len(x)

# Create Gaussians
X_a = (ndarray((N, 2))).T
X_b = (ndarray((N, 2))).T
X_a1, X_a2 = np.random.multivariate_normal(mean_a, covA, N).T
X_b1, X_b2 = np.random.multivariate_normal(mean_b, covB, N).T
X_a[0, :], X_a[1, :] = [X_a1, X_a2]
X_b[0, :], X_b[1, :] = [X_b1, X_b2]

# Visualize
plt.scatter(X_a1, X_a2, c ='b')
plt.scatter(X_b1, X_b2, c ='r')
plt.title('Initial Normal Bivariate Distribution Dataset')

plt.show()

# Plot the dependence of F(w) on the direction of w
fig2, ax1 = plt.subplots(1, 4, figsize=(20, 5))
fig2.suptitle('Outputs given range of weights')
Y_a = []
Y_b = []
weights = np.array([[-0.1, 10, 100, 1000], [-0.1, 10, 100, 1000]])
for i in range(4):
    w = weights[:, i]
    Y_a = w.T.dot(X_a)
    Y_b = w.T.dot(X_b)
    ax1[i].hist(Y_a, 50, color='b', density=True, alpha=0.5)
    ax1[i].hist(Y_b, 50, color='r', density=True, alpha=0.5)
    a = 'w = ' + str(weights[0, i])
    ax1[i].set_title(a)
plt.show()

#  Maximum value of F(w(θ)) and the corresponding direction w∗
u = np.linspace(0, np.pi, 50)
w0 = weights[:, 1]
Fisher = np.zeros((1, len(u)))
Un_Fisher = np.zeros((1, len(u)))
for i, ui in enumerate(u):
    meanA, meanB, devA, devB = [0, 0, 0, 0]
    R = np.array([[np.cos(ui), -np.sin(ui)], [np.sin(ui), np.cos(ui)]])
    w_u = R.dot(w0)
    Y_a = w_u.T.dot(X_a)
    Y_b = w_u.T.dot(X_b)
    for n in range(N):
        meanA += (1 / N) * Y_a[n]
        meanB += (1 / N) * Y_b[n]
    for n in range(N):
        devA += (1 / N) * ((Y_a[n] - meanA) ** 2)
        devB += (1 / N) * ((Y_b[n] - meanB) ** 2)
    Fisher[0, i] = ((meanA - meanB) ** 2) / ((1 / 2) * devA ** 2 + (1 / 2) * devB ** 2)
    Un_Fisher[0, i] = ((meanA - meanB) ** 2) / (devA ** 2 + devB ** 2)
plt.plot(u, Fisher.T, 'b', label='Fisher')
plt.legend()
plt.plot(u, Un_Fisher.T, 'r', label='Unbalanced Fisher')
plt.legend()
plt.xlabel('θ in rad')
plt.ylabel('Fisher Score')
plt.title('F(θ)')
plt.show()
ind_max = np.argmax(Fisher)
F_max = Fisher.T[ind_max]
u_max = u[ind_max]
print('Highest Fisher ratio is %f and occurs at a weight direction of %f rads' % (F_max, u_max))

# Equi-probable contour lines for each class
R_max = np.array([[np.cos(u_max), -np.sin(u_max)], [np.sin(u_max), np.cos(u_max)]])
w_max = R_max.dot(w0)
sns.kdeplot(X_a1, X_a2, cmap='hot')
sns.kdeplot(X_b1, X_b2, cmap='hot')
plt.quiver(w_max[0], w_max[1])
plt.quiver(-w_max[0], -w_max[1])
plt.title('KDE Distributions with optimal direction separation')
plt.show()
mean_a, mean_b, = mean_a.reshape(1, 2), mean_b.reshape(1, 2)

# Baye's Theores and decision boundary
def log_odds_func(x_c):
    x_c = x_c.reshape(N, 2)
    if np.array_equal(covA, covB):
        log_odds = np.log(len(X_a)/len(X_b)) + (mean_a - mean_b).dot(np.linalg.inv(covA)).dot(x_c.T) - \
                   0.5*((mean_a - mean_b).dot(np.linalg.inv(covA)).dot((mean_a - mean_b).T))
    else:
        log_odds = np.log(len(X_a)/len(X_b)) + 0.5*(np.log(np.linalg.det(np.linalg.inv(covA)) /
                                                           np.linalg.det(np.linalg.inv(covB)))) - \
                0.5*x_c.dot(np.linalg.inv(covA) - np.linalg.inv(covB)).dot(x_c.T) +\
                (mean_a - mean_b).dot(np.linalg.inv(covA) - np.linalg.inv(covB)).dot(x_c.T) - \
                0.5*((mean_a - mean_b).dot(np.linalg.inv(covA) - np.linalg.inv(covB)).dot((mean_a - mean_b).T))
    return log_odds

# Plotting
fig6 = plt.figure(figsize=(20, 10))
ax0 = fig6.add_subplot(121)
x_c = np.array([x, x]).T
log_odds1 = log_odds_func(x_c)
ax0.scatter(X_a1, X_a2, c='b')
ax0.scatter(X_b1, X_b2, c='r')
covB = np.array([[10, 0], [0, 10]])
log_odds2 = log_odds_func(x_c)
ax1 = fig6.add_subplot(122, projection='3d')
zeros = np.zeros((len(X_a1), len(X_a2)))
x_c = x_c.reshape(2, 200)
w0n, w1n = np.meshgrid(x_c[0], x_c[1].T)
# ax1.scatter3D(w0n, w1n, X_a, c='b')
# ax1.scatter3D(w0n, w1n, X_b, c='r')
ax0.plot(x_c[0], log_odds1.T)
# ax2[1].projection('3D')
ax1.contour(w0n, w1n, log_odds2)
plt.show()

