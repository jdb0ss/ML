from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.datasets import *
from sklearn.cluster import *
from sklearn.preprocessing import StandardScaler
from sklearn.utils.testing import ignore_warnings
import matplotlib.pyplot as plt

import pickle
with open('trollData.pkl', 'rb') as f:
    trollp = pickle.load(f)
    print("p shape :", trollp.shape)
with open('data.pkl', 'rb') as f:
    params = pickle.load(f)
    print("p shape :", params.shape)

X_troll = trollp
X = np.concatenate([params, X_troll], axis=0)
X = X[:, 1:]
feature_num = len(X[0]) - 1 # feature 갯수
# list to pandas, list to pandas series
panda_X = pd.DataFrame(X)
print("panda_X Shape :", panda_X.shape)

print("X_dataShape : ", X.shape)
print("T_dataShape : ", X_troll.shape)
data = StandardScaler().fit_transform(X)  # normalize 기능
print("DataShape : ", data.shape)
n_samples, n_features = data.shape
print("n_sample :", n_samples, end=" / ")
print("n_features :", n_features)

pca = PCA(n_components=feature_num)
pca.fit(X)
print("singular value :", pca.singular_values_)
print("singular vector :\n", pca.components_.T)
print("eigen_value :", pca.explained_variance_)
print("explained variance ratio :", pca.explained_variance_ratio_)
pca = PCA(n_components=0.99999)
X_reduced = pca.fit_transform(X)
print("X_redu Shape : ", X_reduced.shape)
print("explained variance ratio :", pca.explained_variance_ratio_)
print("선택한 차원수 :",pca.n_components_)
print(X_reduced.shape)

plt.figure(figsize=(9, 8))
n_clusters = 6            # 클러스터 수

X_reduced = StandardScaler().fit_transform(X_reduced)
X = StandardScaler().fit_transform(X)
two_means = MiniBatchKMeans(n_clusters=n_clusters)
dbscan = DBSCAN(eps=0.6)  # 밀도 기반 클러스터링
spectral = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors")
ward = AgglomerativeClustering(n_clusters=n_clusters)
affinity_propagation = AffinityPropagation(damping=0.9, preference=-200)    # 매우 느림
clustering_algorithms = (
    ('K-Means', two_means),
    ('DBSCAN', dbscan),
    ('Hierarchical Clustering', ward),
    ('Spectral Clustering', spectral),
    # ('Affinity Propagation', affinity_propagation),
)

plot_num = 1

for j, (name, algorithm) in enumerate(clustering_algorithms):
    with ignore_warnings(category=UserWarning):
        algorithm.fit(X_reduced)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X_reduced)
    y_pred += 1
    plt.subplot(len(clustering_algorithms), 2, plot_num)
    print("y_pred :", y_pred)
    print("len :", len(y_pred))
    y_max = np.max(y_pred)
    print("maxnum :", np.max(y_pred))
    colors = plt.cm.tab10(np.arange(20, dtype=int))
    # s 포인트 크기, color 배열
    y_reduced = y_pred[:10000]
    y_troll = []
    for i in range (0, len(X_troll)):
        y_troll.append(0)
    plt.scatter(X_reduced[:10000,0], X_reduced[:10000,1], s=2, color=colors[y_reduced])
    plt.scatter(X_reduced[10000:,0], X_reduced[10000:,1], s=2, color=colors[y_troll])
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