# author : Lee
# date   : 2021/6/27 16:45

import numpy as np
import numpy.linalg as LA
from scipy.stats import anderson
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class GMeans:
    def __init__(self, alpha=4, n_init_clusters=1, n_init=10, random_state=None, use_pca=False, cluster_min_size=8):
        """
        :param alpha: significance level
        :param n_init_clusters: the initial number of clusters
        :param n_init: number of time the k-means algorithm will be run with different centroid seeds
        :param random_state: determines random number generation for centroid initialization
        :param use_pca: use PAC
        :param cluster_min_size: minimum number of data points in a cluster
        """
        if alpha not in range(5):
            raise ValueError("Parameter alpha must be integer from 0 to 4")
        self.n_init_clusters = n_init_clusters
        self.alpha = alpha
        self.n_init = n_init
        self.random_state = random_state
        self.use_pca = use_pca
        self.cluster_min_size = cluster_min_size

    def ad_test(self, x):
        """
        Anderson-Darling test for data coming from a Gaussian distribution.
        H0: X obeys Gaussian distribution
        H1: X does not obey Gaussian distribution
        :param x: array of sample data.
        :return: True if accept H0
        """
        res = anderson(x)
        if res.statistic < res.critical_values[self.alpha]:
            return True
        return False

    def g_means(self, X, index, label):
        """
        :param X: (n_samples, n_features) sub dataset
        :param index: index of elements in X in entire data set
        :param label: label of X
        Gaussian-means algorithm
        :return:
        """
        if X.shape[0] < self.cluster_min_size:
            return
        kmeans = KMeans(n_clusters=2,
                        n_init=self.n_init,
                        random_state=self.random_state).fit(X)
        if self.use_pca:
            X_ = (PCA(n_components=1).fit(X).components_ @ X.T)[0]  # PAC
        else:
            c = kmeans.cluster_centers_
            v = c[0] - c[1]
            X_ = X.dot(v) / LA.norm(v)  # project data points to v
            X_ = X.dot(v) / v.dot(v)  # project data points to v
            X_ = (X_ - np.mean(X_)) / np.std(X_)
        H0 = self.ad_test(X_)
        if H0:
            return
        self.cluster_centers_[label] = kmeans.cluster_centers_[0]
        self.cluster_centers_ = np.vstack((self.cluster_centers_, kmeans.cluster_centers_[1]))
        self.labels_[index[kmeans.labels_ == 1]] = self.n_cur_clusters
        self.n_cur_clusters += 1
        self.g_means(X[kmeans.labels_ == 1], index[kmeans.labels_ == 1], self.n_cur_clusters - 1)
        self.g_means(X[kmeans.labels_ == 0], index[kmeans.labels_ == 0], label)

    def fit(self, X):
        """
        Compute G-means clustering
        :param X: (n_samples, n_features) entire dataset
        :return: self
        """
        # FIG_2 ###########
        kmeans = KMeans(n_clusters=self.n_init_clusters,
                        n_init=self.n_init,
                        random_state=self.random_state).fit(X)
        self.n_cur_clusters = self.n_init_clusters
        self.labels_ = kmeans.labels_
        self.cluster_centers_ = kmeans.cluster_centers_
        for label in range(self.n_init_clusters):
            self.g_means(X[self.labels_ == label], np.argwhere(self.labels_ == label), label)
        self.n_clusters = self.n_cur_clusters
        return self


def estimate(n_repeat=1, n_samples=5000, n_features=2, n_centers=4, cluster_std=0.5, random_state=0, use_pca=False):
    """
    :param n_repeat: Repeat times
    :param n_samples: The total number of points equally divided among clusters
    :param n_features: The number of features for each sample.
    :param n_centers: The number of centers to generate
    :param cluster_std: The standard deviation of the clusters.
    :param random_state: Determines random number generation for dataset creation.
    :return:
    """
    all_n_clusters = []
    for _ in range(n_repeat):
        # Generate isotropic Gaussian blobs for clustering.
        X, y, centers = datasets.make_blobs(n_samples=n_samples, n_features=n_features, return_centers=True,
                                            centers=n_centers, cluster_std=cluster_std, random_state=None)
        np.random.seed(random_state)
        trans = np.random.uniform(0.5, 2, n_features)
        centers = centers * trans
        X = X * trans  # Make a random transformation in each dimension of the data
        gmeans = GMeans(random_state=random_state, use_pca=use_pca).fit(X)
        all_n_clusters.append(gmeans.n_clusters)
    all_n_clusters = np.array(all_n_clusters)
    return np.mean(all_n_clusters), np.std(all_n_clusters), np.min(all_n_clusters), np.max(all_n_clusters)


def main():
    repeat = 50
    samples = 5000
    features = [2, 4, 8, 16, 32, 64, 128]
    centers = [4, 16, 64, 128]
    std = 0.5
    random_state = 1
    print('d' + ' ' * 6 + 'k' + ' ' * 15 + 'G-means' + ' ' * 18 + 'G-means(PCA)')
    print(' ' * 15 + 'avg   std   min   max' + ' ' * 7 + 'avg   std   min   max')
    for d in features:
        for i in range(len(centers)):
            k = centers[i]
            gmeans = estimate(n_repeat=repeat, n_centers=k, n_features=d,
                              cluster_std=std, random_state=random_state, use_pca=False)
            gmeans_pca = estimate(n_repeat=repeat, n_centers=k, n_features=d,
                                  cluster_std=std, random_state=random_state, use_pca=True)
            print('{:<7}{:<8}{:<6.1f}{:<6.1f}{:<6}{:<10}{:<6.1f}{:<6.1f}{:<6}{:<6}'.
                  format(d, k, gmeans[0], gmeans[1], gmeans[2], gmeans[3],
                         gmeans_pca[0], gmeans_pca[1], gmeans_pca[2], gmeans_pca[3]))


if __name__ == '__main__':
    main()
