import numpy as np
from sklearn.datasets import load_iris
import random
from logclass import LOGISTIC    # LOGISTICclass import

iris = load_iris()
X = iris.data[:, :4]        # feature 4종류.
y = iris.target             # 0, 1, 2 로 구성되어있는 데이터
y_name = iris.target_names  # ['setosa' 'versicolor' 'virginica'] 꽃 이름

bias = []                   # bias를 추가하기 위한 리스트
for i in range(0, len(X)):  # data set 크기 만큼 1을 담고 있는 bias 리스트 생성
    bias.append([1])
bias = np.array(bias)       # np.array bias : shape (150, 1)
X = np.array(X)             # np.array X : shape (150, 4)
X = np.hstack((bias, X))    # column으로 np.array를 합침 X : shape (150, 5)

# one-hot encoding
num = np.unique(y, axis=0)  # [0 1 2]
num = num.shape[0]          # 3
y = np.eye(num)[y]          # y = [[1. 0. 0.] [1. 0. 0.] ... [0. 0. 1.]]

l = 5                       # train data set size : test data set size = 8 : 2
# y.shape[0] = 150 이며 i % l == 5 일때를 test data set 으로 설정
for_test = np.array([(i % l == (l-1)) for i in range(y.shape[0])]) # for_test 는 boolean 값을 담는 리스트가 된다.
for_train = ~for_test       # for_train 은 for_test 의 반대 boolean 값을 담는 리스트가 된다.
X_train = X[for_train]      # iris.data X에서 for_train 의 true 값만을 담는 리스트가 된다. 크기 120 train data set 의 feature
y_train = y[for_train]      # iris.target y에서 for_train 의 true 값만을 담는 리스트가 된다. 크기 120 train data set 의 답
X_test = X[for_test]        # iris.data X에서 for_test 의 true 값만을 담는 리스트가 된다. 크기 30 test data set 의 feature
y_test = y[for_test]        # iris.target y에서 for_test 의 true 값만을 담는 리스트가 된다. 크기 30 test data set 의 답

weight = []    # multi classification weight 를 담고 있는 2차원 배열
for i in range(0, len(y[0])):    # class 종류 수 만큼 for 문
    w_n = []                     # class i번째에 대한 weight 리스트
    for j in range(0, len(X[0])):   # data feature 수 만큼 for 문
        w_n.append(random.random()) # 초기 weight 값은 0~1 사이의 random 값으로 초기화
    weight.append(w_n)           # class i번째에 대한 weight 리스트 append
weight = np.array(weight)        # np.array weight : shape (3, 5)

epoch = 10000  # 학습 횟수
# binary classification
for i in range(0, weight.shape[0]): # class 종류 수 만큼 for 문
    single_iris = LOGISTIC(X_train, y_train[:, i], weight[i, :], X_test, y_test[:, i], y_name[i]) # i 번째 single class
    # (train data feature, train data answer[i], weight[i], test data feature, test data answer[i], answer_name[i])
    for j in range(0, epoch):   # 학습 횟수 만큼 for 문
        single_iris.cost()      # cost 함수 계산
        single_iris.learn()     # 학습 (modify weight가 일어난다, gradient descent 과정)
    single_iris.predict()       # 완성된 weight로 test data 결과 예측, 정확도 출력
    single_iris.printgrap()     # 그래프 출력 함수 실행
# multi classification
multi_iris = LOGISTIC(X_train, y_train, weight, X_test, y_test, y_name)  # multi class
for i in range(0, epoch):       # 학습 횟수 만큼 for 문
    multi_iris.cost()           # cost 함수 계산
    multi_iris.learn()          # 학습 (modify weight가 일어난다, gradient descent 과정)
multi_iris.predict()            # 완성된 weight로 test data 결과 예측, 정확도 출력
multi_iris.printgrap()          # 그래프 출력 함수 실행