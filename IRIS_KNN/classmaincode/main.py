import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from knnclass import KNN    # knnclass import

iris = load_iris()
X = iris.data[:, :4]        # feature 4종류.
y = iris.target             # 0, 1, 2 로 구성되어있는 데이터
y_name = iris.target_names  # ['setosa' 'versicolor' 'virginica'] 꽃 이름

l = 15
# y.shape[0] = 150 이며 i % l == 14 일때를 test data set 으로 설정한다.
for_test = np.array([(i % l == (l-1)) for i in range(y.shape[0])]) # for_test 는 boolean 값을 담는 리스트가 된다.
for_train = ~for_test       # for_train 은 for_test 의 반대 boolean 값을 담는 리스트가 된다.
# print(len(X_train)) = 140 , print(len(y_train)) = 140
X_train = X[for_train]      # iris.data X에서 for_train 의 true 값만을 담는 리스트가 된다. 크기 140 train data set 의 feature
y_train = y[for_train]      # iris.target y에서 for_train 의 true 값만을 담는 리스트가 된다. 크기 140 train data set 의 답
# print(len(X_test)) = 10, print(len(y_test)) = 10
X_test = X[for_test]        # iris.data X에서 for_test 의 true 값만을 담는 리스트가 된다. 크기 10 test data set 의 feature
y_test = y[for_test]        # iris.target y에서 for_test 의 true 값만을 담는 리스트가 된다. 크기 10 test data set 의 답

knn_iris = KNN(10, X_train, y_train, y_name)    # KNN class 객체 생성 (k, train set feature, train set 답, 꽃 이름 리스트)
knn_iris.show_dim()            # feature 갯수 출력 (dimension 과 동일 = 4)
klist = [3, 5, 10]             # k = 3, 5, 10 에 대해 돌려주기 위한 리스트
for j in range (0, 3):         # klist 인자로 k = 3, 5, 10 을 돌려주기 위한 for 문
    knn_iris.set_k(klist[j])   # class 객체의 set_k (k 값 설정)

    # Majority Vote
    print("Majority vote / k = ", klist[j])
    for i in range(y_test.shape[0]):        # test data set 에 대한 for 문
        knn_iris.get_nearest_k(X_test[i])   # X_test를 하나씩 넣으며, 가장 가까운 point 10개를 얻고 객체의 리스트에 저장한다.
        # Test Data i 번째 결과를 majority vote 를 통해 출력 , 실제 답(꽃이름)을 y_name[y_test[i]] 으로 출력
        print("Test Data", i, "Computed class:", knn_iris.mv(), ",\tTrue class: ", y_name[y_test[i]])
        knn_iris.reset()                    # 가장 가까운 순으로 정렬된 point 리스트 초기화, majority vote 리스트 초기화

    # test data set 10 개를 모두 실행하면 결과를 저장한 grapcor 리스트를 get_grapcor 을 통해 호출하여 그래프를 출력한다.
    plt.figure(4, figsize=(8, 6))  # matplotlib 출력 화면 크기 지정
    plt.scatter(X_test[:, 0], X_test[:, 1], c=knn_iris.get_grapcor(), cmap=plt.cm.Set1, edgecolor='k')  # 0 수평 1 수직
    plt.title("majority vote / K = " + str(klist[j]))   # 그래프 제목 (알고리즘 / k 값)
    plt.xlabel('Sepal length')  # xlabel 이름
    plt.ylabel('Sepal width')   # ylabel 이름
    plt.xticks(())
    plt.yticks(())
    plt.show()                  # 그래프 출력
    knn_iris.reset_grapcor()    # grapcor 리스트 초기화

    # Weighted Majority Vote
    print("Weigthed Majority vote / k = ", klist[j])
    for i in range(y_test.shape[0]):        # test data set 에 대한 for 문
        knn_iris.get_nearest_k(X_test[i])   # X_test를 하나씩 넣으며, 가장 가까운 point 10개를 얻고 객체의 리스트에 저장한다.
        # Test Data i 번째 결과를 weighted majority vote 를 통해 출력 , 실제 답(꽃이름)을 y_name[y_test[i]] 으로 출력
        print("Test Data", i, "Computed class:", knn_iris.wmv(), ",\tTrue class: ", y_name[y_test[i]])
        knn_iris.reset()                    # 가장 가까운 순으로 정렬된 point 리스트 초기화, majority vote 리스트 초기화

    # test data set 10 개를 모두 실행하면 결과를 저장한 grapcor 리스트를 get_grapcor 을 통해 호출하여 그래프를 출력한다.
    plt.figure(4, figsize=(8, 6))  # matplotlib 출력 화면 크기 지정
    plt.scatter(X_test[:, 0], X_test[:, 1], c=knn_iris.get_grapcor(), cmap=plt.cm.Set1, edgecolor='k')  # 0 수평 1 수직
    plt.title("weighted majority vote / K = " + str(klist[j])) # 그래프 제목 (알고리즘 / k 값)
    plt.xlabel('Sepal length')  # xlabel 이름
    plt.ylabel('Sepal width')   # ylabel 이름
    plt.xticks(())
    plt.yticks(())
    plt.show()                  # 그래프 출력
    knn_iris.reset_grapcor()    # grapcor 리스트 초기화