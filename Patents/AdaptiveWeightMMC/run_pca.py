import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.decomposition import PCA

database = 'coil'
if database == 'orl':
    path = '../../Resources/OrlDataset/ORL_32x32.mat'
    dim = np.arange(10, 120, 10)
    alpha = 1
    neighbor = 4
elif database == 'yale':
    path = '../../Resources/YaleDataset/Yale_32x32.mat'
    dim = np.arange(1, 115, 3)
    alpha = 1
    neighbor = 4
elif database == 'coil':
    path = '../../Resources/Coil20Database/COIL20.mat'
    dim = np.arange(10, 410, 10)
    alpha = 1
    neighbor = 4

data = scio.loadmat(path)

X = data['fea']
gnd = data['gnd']

repeat = 10
accs = np.zeros([repeat, len(dim)])

for i in range(dim.shape[0]):
    for j in range(accs.shape[0]):
        train_X, test_X, train_gnd, test_gnd = train_test_split(X, gnd, test_size=0.3, random_state=j,
                                                                stratify=gnd)
        mu = np.mean(train_X, 0).reshape(-1, 1).T
        train_X = train_X - mu
        test_X = test_X - mu

        pca = PCA(n_components=dim[i], whiten=False)
        pca.fit(train_X)
        train_X = pca.transform(train_X)
        test_X = pca.transform(test_X)
        clf = neighbors.KNeighborsClassifier(n_neighbors=neighbor)
        clf.fit(train_X, train_gnd.flatten())
        accs[j, i] = clf.score(test_X, test_gnd.flatten())
        # label = clf.predict(test_X @ eig_vec)
        # a = np.mean(label == test_gnd.flatten())
accuracy = np.mean(accs, 0)
