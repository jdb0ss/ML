import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# from knn (file명) import KNN (class명)

iris = load_iris()
print(iris)
# for now, use the first two features.
# feature 를 두개 쓰겠다
X = iris.data[:, :2]
y = iris.target

# l = 15
# for_test = np.array([(i%l==(l-1)) for i in range(y.shape[0])])
# for_train = ~for_test
# X_train = X[for_train]
# Y_train = Y[for_train]
# print(for_test)

#X_test = X[for_test]
#Y_test = y[for_test]
# print(X_test)
# print(Y_train)
# print(y_name_train)

# knn_iris = KNN(10, X_train, y_train, y_name)
# knn_iris.show_dim()

# for i in range(y_test.shape[0]):
#     knn_iris.get_nearest_k(X_test[i])
#     print("Test Data", i, "Computed class:", knn_iris.wmv()),
#     ",\tTrue class: ", y_name[y_test[i]]
#     knn_iris.reset()


# x1 min max 정해주고, x2 min max 정해준다
x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
plt.figure(2, figsize=(8, 6))  # 화면 크기 지정

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')  # 0 수평 1 수직
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.xticks(())
plt.yticks(())
plt.show()

# 변수 K, Features : X , Target : y
# 기능 distance 계산, k 개의 가장 가까운 것 찾기, majority vote weighted majority vote
# 이거 함수 클래스화 해야함
# 14개 까지가 train , 15 번째가 test 로 쓴다
# Train : every 1st ... 14th data [0] ... [13]
# Test : 15th data[14] , data[29] ...
# input K 3종류 * output m.v , w.mv 로 = 총 6개의 결과
# knn 클래스.py + 아이리스 데이터 불러오고 knn 쓰는 .py
# wegiht = 1/distance > 문제점 거리가 아주가까우면 걍 맛탱이감
