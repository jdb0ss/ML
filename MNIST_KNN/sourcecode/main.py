import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
from knnclass import KNN    # knnclass import

# training data, test data,
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
# flattern : 이미지를 1차원 배열로 읽음 , normalize : 0~1 실수로. 그렇지 않으면 0~255

size = 100
sample = np.random.randint(0, t_test.shape[0], size)

label_name =  ['0','1','2','3','4','5','6','7','8','9']
mnist_allfeature = KNN(10, x_train, t_train, label_name)   # KNN class 객체 생성 Use All Features = 784
mnist_handcraft = KNN(10, x_train, t_train, label_name)    # KNN class 객체 생성 Compression Features = 102
mnist_handcraft.handcraft_feature()                        # Train data set Features 변환
klist = [3, 5, 10]     # k = 3, 5, 10 에 대해 돌려주기 위한 리스트
# Use All Features 784
output = np.zeros((size,3,2))
acculist = np.zeros((3,2))
for i in range(0, size):  # Test data set 에 대한 for 문
    mnist_allfeature.get_nearest_k(x_test[sample[i]], 'Use all Features')  # x_test를 하나씩 넣으며 거리기준 disarr 리스트 완성
    for j in range(0, 3):  # klist 인자로 k = 3, 5, 10 을 돌려주기 위한 for 문
        mnist_allfeature.set_k(klist[j])
        (computed, trueclass) = (mnist_allfeature.mv(), label_name[t_test[sample[i]]])
        mnist_allfeature.rest_votecnt()  # vote_cnt 리스트만 초기화, disarr 리스트는 그대로
        output[i][j][0] = computed
        if computed == trueclass:       # 계산결과 == 실제답 이면
            acculist[j][0] += 1 / size  # 정확도 1 / size 만큼 증가
        (computed, trueclass) = (mnist_allfeature.wmv(), label_name[t_test[sample[i]]])
        mnist_allfeature.rest_votecnt()  # vote_cnt 리스트만 초기화, disarr 리스트는 그대로
        output[i][j][1] = computed
        if computed == trueclass:       # 계산결과 == 실제답 이면
            acculist[j][1] += 1 / size  # 정확도 1 / size 만큼 증가
    mnist_allfeature.reset()  # 가장 가까운 순으로 정렬된 point 리스트 초기화, majority vote 리스트 초기화

for i in range(0, len(klist)):
    print("Use all Features Majority vote / k = ", klist[i])
    for j in range(0, size):
        print("TestData", j, "index(", sample[j], ") Computedclass:", int(output[j][i][0]), ",\tTrueclass: ", label_name[t_test[sample[j]]])
    print("Accuracy :", round(acculist[i][0], 3))
    print("Use all Features, Weighted Majority vote / k = ", klist[i])
    for j in range(0, size):
        print("TestData", j, "index(", sample[j], ") Computedclass:", int(output[j][i][1]), ",\tTrueclass: ", label_name[t_test[sample[j]]])
    print("Accuracy :", round(acculist[i][1], 3))

# Features Compression 112
output = np.zeros((size,3,2))
acculist = np.zeros((3,2))
for i in range(0, size):  # Test data set 에 대한 for 문
    mnist_handcraft.get_nearest_k(x_test[sample[i]], 'feature_compression')  # x_test를 하나씩 넣으며 거리기준 disarr 리스트 완성
    for j in range(0, 3):  # klist 인자로 k = 3, 5, 10 을 돌려주기 위한 for 문
        mnist_handcraft.set_k(klist[j])
        (computed, trueclass) = (mnist_handcraft.mv(), label_name[t_test[sample[i]]])
        mnist_handcraft.rest_votecnt()  # vote_cnt 리스트만 초기화, disarr 리스트는 그대로
        output[i][j][0] = computed
        if computed == trueclass:       # 계산결과 == 실제답 이면
            acculist[j][0] += 1 / size  # 정확도 1 / size 만큼 증가
        (computed, trueclass) = (mnist_handcraft.wmv(), label_name[t_test[sample[i]]])
        mnist_handcraft.rest_votecnt()  # vote_cnt 리스트만 초기화, disarr 리스트는 그대로
        output[i][j][1] = computed
        if computed == trueclass:       # 계산결과 == 실제답 이면
            acculist[j][1] += 1 / size  # 정확도 1 / size 만큼 증가
    mnist_handcraft.reset()  # 가장 가까운 순으로 정렬된 point 리스트 초기화, majority vote 리스트 초기화

for i in range(0, len(klist)):
    print("Features Compression, Majority vote / k = ", klist[i])
    for j in range(0, size):
        print("TestData", j, "index(", sample[j], ") Computedclass:", int(output[j][i][0]), ",\tTrueclass: ", label_name[t_test[sample[j]]])
    print("Accuracy :", round(acculist[i][0], 3))
    print("Features Compression, Weighted Majority vote / k = ", klist[i])
    for j in range(0, size):
        print("TestData", j, "index(", sample[j], ") Computedclass:", int(output[j][i][1]), ",\tTrueclass: ", label_name[t_test[sample[j]]])
    print("Accuracy :", round(acculist[i][1], 3))