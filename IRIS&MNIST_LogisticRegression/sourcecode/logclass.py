import numpy as np
import matplotlib.pyplot as plt
class LOGISTIC():
    def __init__(self, x_input, y_output, weight, x_test, y_test, labelname, learning_rate = 0.01, cyclelen = 20):
        self.x_input = x_input              # train data set - feature
        self.y_output = y_output            # train data set - answer
        self.weight = weight                # 학습과정에서 계속 수정되는 가중치
        self.x_test = x_test                # test data set - feature
        self.y_test = y_test                # test data set - answer
        self.learning_rate = learning_rate  # learning rate
        self.cyclelen = cyclelen            # cost 출력 빈도 (단위)
        self.hx = self.sigmoid(np.dot(self.x_input, self.weight.transpose()))   # hypothesis 값 초기화
        self.costrr = []                    # cost 값을 가지고 있는 리스트
        self.dim = x_input[0].__len__()     # feature 갯수 (dimension)
        self.xsize = len(x_input)           # train data set size
        self.epoch = 0                      # epoch 값 (현재 학습 횟수)
        self.grap_costrr = []               # 그래프 출력을 위해 cost 값을 저장하는 리스트
        self.grap_epoch = []                # 그래프 출력을 위한 epoch 값을 저장하는 리스트
        self.grap_legend = labelname        # multi classification 에서 각 그래프에 이름을 표시하기 위한 legend 리스트
        if y_test.ndim == 1:                # binary classification 이라면
            self.grap_title = 'binary classification : ' + labelname     # 그래프 제목은 binary ... labelname
        else:                               # multi classification 이라면
            self.grap_title = 'multi classification'                     # 그래프 제목은 multi classification

    def sigmoid(self, x):                   # 시그모이드 함수
        eMin = -np.log(np.finfo(type(0.1)).max) # -> -709.78 ...
        xsafe = np.array(np.maximum(x, eMin))   # exp 값 즉, e^x 가 너무 커지는 것을 방지하기 위해 최대값 상정
        return 1 / (1+np.exp(-xsafe))           # 시그모이드 함수 공식 적용후 return

    def cost(self):                         # cost 함수
        # J(θ) = -1/m[Σy(i)log hθ(x(i)) + (1-y(i))log(1-hθ(x(i))] 에 해당하는 코드
        self.costrr = -1 * sum(self.y_output * np.log(self.hx) + (1-self.y_output) * np.log(1-self.hx)) / self.xsize
        self.grap_costrr.append(self.costrr) # 계산한 cost 값은 그래프 출력을 위해 저장

    def learn(self):                        # 학습 1회에 해당하는 함수
        self.grap_epoch.append(self.epoch)  # 그래프 출력을 위해 epoch 저장
        if self.epoch % self.cyclelen == 0: # 출력 단위가 되었으면
            print("epoch :", self.epoch, "cost :", self.costrr) # epoch 와 cost 를 출력한다.
        self.hx = self.sigmoid(np.dot(self.x_input, self.weight.transpose())) # modify hypothesis
        dif = self.hx - self.y_output                                         # hθ(x(i)) - y(i)
        gradient = np.dot(self.x_input.transpose(), dif) / self.xsize         # Σ(hθ(x(i)) - y(i))xj(i)
        self.weight = self.weight - self.learning_rate * gradient.transpose() # θj = θj - Σ(hθ(x(i)) - y(i))xj(i)
        self.epoch += 1 # 학습 횟수 1 증가

    def predict(self):             # 완성된 학습으로 test data set에 대해 결과를 예측하는 함수
        accuracy = 0.              # 정확도
        if  self.y_test.ndim == 1: # binary classification 경우
            hypo = self.sigmoid(np.dot(self.x_test, self.weight.transpose()))  # test data로 hypothesis 값 계산
            for i in range (0, len(self.x_test)):           # for 문 test data 크기 만큼
                if hypo[i] > 0.5 and self.y_test[i] == 1:   # hypothesis가 0.5 이상이면 결과 1로 상정, 답을 맞췄을 경우
                    accuracy += (1 / len(self.x_test))      # 정확도 += 1 / test data 크기
                if hypo[i] <= 0.5 and self.y_test[i] == 0:  # hypothesis가 0.5 이하이면 결과 0로 상정, 답을 맞췄을 경우
                    accuracy += (1 / len(self.x_test))      # 정확도 += 1 / test data 크기
        else:                      # multi classification 경우
            hypo = self.sigmoid(np.dot(self.x_test, self.weight.transpose()))  # test data로 hypothesis 값 계산
            for i in range (0, len(self.x_test)):           # for 문 test data 크기 만큼
                if np.argmax(hypo[i]) == np.argmax(self.y_test[i]): # class 중 hypothesis 값이 가장큰 것을 결과로 상정
                    accuracy += (1 / len(self.x_test))      # argmax로 값이 가장큰 인덱스 추출, 같으면 답을 맞췄을 경우이다
        print("Accuracy :", round(accuracy, 3)) # 계산한 정확도 소숫점 3자리까지 출력

    def printgrap(self):    # 결과에 대한 그래프를 출력하는 함수
        plt.plot(self.grap_epoch, self.grap_costrr) # x축 epoch, y축 cost값
        plt.ylim(0.005, 0.4)          # 보다 직관적인 결과 관찰을 위해 y축 확대
        plt.xlabel('epoch')           # xlabel : epoch
        plt.ylabel('cost')            # ylabel : cost
        plt.title(self.grap_title)    # 그래프 제목 설정
        plt.legend(self.grap_legend)  # 그래프 각각에 대해 legend 표시
        plt.show()                    # 출력