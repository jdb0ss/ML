from sklearn.datasets import *
from sklearn.cluster import *
from sklearn.preprocessing import StandardScaler
from sklearn.utils.testing import ignore_warnings
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('data.pkl', 'rb') as f:
    params = pickle.load(f)
    print("p shape :", params.shape)

X_reduced = params
np.random.seed(0)
n_samples = 10000

# iris = load_iris()
# X = iris.data[:, :4]        # feature 4종류.
# y = iris.target             # 0, 1, 2 로 구성되어있는 데이터
print("data shape :", X_reduced.shape)

plt.figure(figsize=(10, 10))
n_clusters = 9

X_reduced = StandardScaler().fit_transform(X_reduced)
two_means = MiniBatchKMeans(n_clusters=n_clusters)
dbscan = DBSCAN(eps=0.15)  # 밀도 기반 클러스터링
spectral = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors")
ward = AgglomerativeClustering(n_clusters=n_clusters)
affinity_propagation = AffinityPropagation(damping=0.9, preference=-200)
clustering_algorithms = (
    ('K-Means', two_means),
    # ('DBSCAN', dbscan),
    # ('Hierarchical Clustering', ward),
    # ('Affinity Propagation', affinity_propagation),
    # ('Spectral Clustering', spectral),
)

plot_num = 1

for j, (name, algorithm) in enumerate(clustering_algorithms):
    with ignore_warnings(category=UserWarning):
        algorithm.fit(X_reduced)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X_reduced)
    plt.subplot(len(clustering_algorithms), len(clustering_algorithms), plot_num)

    colors = plt.cm.tab10(np.arange(40, dtype=int))
    # print("colorlen: ", len(colors)) = 20
    # print(colors.shape) = (20, 4)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=5, color=colors[y_pred])
    # plt.xlim(-2.5, 2.5)
    # plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plot_num += 1

plt.tight_layout()
plt.show()

ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(X_reduced)
    inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()