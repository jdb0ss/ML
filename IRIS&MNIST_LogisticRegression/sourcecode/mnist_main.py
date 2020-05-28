import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import random
from PIL import Image
from logclass import LOGISTIC    # LOGISTICclass import
# training data, test data,
(X, y), (x_test, y_test) = load_mnist(flatten=True, normalize=True)
# flattern : 이미지를 1차원 배열로 읽음 , normalize : 0~1 실수로. 그렇지 않으면 0~255
bias = []                   # bias를 추가하기 위한 리스트
for i in range(0, len(X)):  # data set 크기 만큼 1을 담고 있는 bias 리스트 생성
    bias.append([1])
bias = np.array(bias)       # np.array bias : shape (60000, 1)
X = np.array(X)             # np.array X : shape (60000, 784)
X = np.hstack((bias, X))    # column으로 np.array를 합침 X : shape (60000, 785)
num = np.unique(y, axis=0)  # [0 1 2 ... 8 9]
num = num.shape[0]          # 10
y = np.eye(num)[y]          # y = [[0 0 0 ... 0 0 0] [1 ... 0] ... [0 0 0 ... 0 1 0]]
bias = np.array(bias[:len(x_test)]) # np.array bias : shape (10000, 1)
x_test = np.array(x_test)           # np.array x_test : shape (10000, 784)
x_test = np.hstack((bias, x_test))  # column으로 np.array를 합침 x_test : shape (10000, 785)
y_test = np.eye(num)[y_test] # y_test = [[0 0 0 ... 0 0 0] [1 ... 0] ... [0 0 0 ... 0 1 0]]

weight = []   # multi classification weight 를 담고 있는 2차원 배열
for i in range(0, len(y[0])): # class 종류 수 만큼 for 문
    w_n = []                  # class i번째에 대한 weight 리스트
    for j in range(0, len(X[0])):         # data feature 수 만큼 for 문
        w_n.append(random.random() / 100) # 초기 weight 값은 0/100 ~ 1/100 사이의 값
    weight.append(w_n)        # class i번째에 대한 weight 리스트 append
weight = np.array(weight)     # np.array weight : shape (10, 785)

epoch = 5000  # 학습 횟수
label_name =  ['0','1','2','3','4','5','6','7','8','9'] # 출력을 위한 target name
# single classification
for i in range(0, weight.shape[0]): # class 종류 수 만큼 for 문
    single_mnist = LOGISTIC(X, y[:, i], weight[i, :], x_test, y_test[:, i], label_name[i]) # i 번째 single class
    for j in range(0, epoch):   # 학습 횟수 만큼 for 문
        single_mnist.cost()     # cost 함수 계산
        single_mnist.learn()    # 학습 (modify weight가 일어난다, gradient descent 과정)
    single_mnist.predict()      # 완성된 weight로 test data 결과 예측, 정확도 출력
    single_mnist.printgrap()
# multi classification
multi_mnist = LOGISTIC(X, y, weight, x_test, y_test, label_name) # multi class
for i in range(0, epoch):       # 학습 횟수 만큼 for 문
    multi_mnist.cost()          # cost 함수 계산
    multi_mnist.learn()         # 학습 (modify weight가 일어난다, gradient descent 과정)
multi_mnist.predict()           # 완성된 weight로 test data 결과 예측, 정확도 출력
multi_mnist.printgrap()         # 그래프 출력 함수 실행