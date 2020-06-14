import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from func import weightmmc

database = 'coil'
if database == 'orl':
    path = '../../Resources/OrlDataset/ORL_32x32.mat'
    dim = 50
    alpha = 1
elif database == 'yale':
    path = '../../Resources/YaleDataset/Yale_32x32.mat'
    dim = 50
    alpha = 0
elif database == 'coil':
    path = '../../Resources/Coil20Database/COIL20.mat'
    dim = 50
    alpha = 1

data = scio.loadmat(path)

X = data['fea']
gnd = data['gnd']


train_X, test_X, train_gnd, test_gnd = train_test_split(X, gnd, test_size=0.3, random_state=0, stratify=gnd)
mu = np.mean(train_X, 0).reshape(-1, 1).T
train_X = train_X - mu
test_X = test_X - mu

eig_value, eig_vec, obj = weightmmc(train_X, train_gnd, dim, alpha, Iter=20)
print(obj[:, 1].reshape(-1,1))
