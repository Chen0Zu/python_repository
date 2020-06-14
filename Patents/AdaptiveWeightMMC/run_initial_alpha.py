import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from func import weightmmc

# path = '../../Resources/OrlDataset/ORL_32x32.mat'
# path = '../../Resources/YaleDataset/Yale_32x32.mat'
# path = '../../Resources/YaleExtendedDataset/YaleB_32x32.mat'
# dim = 50

path = '../../Resources/Coil20Database/COIL20.mat'
dim = 30

data = scio.loadmat(path)

X = data['fea']
gnd = data['gnd']


repeat = 10
alphas = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]).reshape(-1, 1)
accs = np.zeros([repeat, len(alphas)])

for i in range(accs.shape[1]):
    for j in range(accs.shape[0]):
        train_X, test_X, train_gnd, test_gnd = train_test_split(X, gnd, test_size=0.3, random_state=j, stratify=gnd)
        mu = np.mean(train_X, 0).reshape(-1, 1).T
        train_X = train_X - mu
        test_X = test_X - mu

        eig_value, eig_vec, obj = weightmmc(train_X, train_gnd, dim, alphas[i], Iter=10)
        clf = neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(train_X @ eig_vec, train_gnd.flatten())
        accs[j, i] = clf.score(test_X @ eig_vec, test_gnd.flatten())
        # label = clf.predict(test_X @ eig_vec)
        # a = np.mean(label == test_gnd.flatten())
accuracy = np.mean(accs, 0)
