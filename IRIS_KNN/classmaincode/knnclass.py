from operator import itemgetter # 리스트 sorting 시 특정 key 값으로 정렬하기 위함

class KNN():
    def __init__(self, k, xtrain, ytrain, yname):
        self.k = k                                  # KNN 에서의 k 값
        self.xtrain = xtrain                        # train data set feature
        self.ytrain = ytrain                        # train data set 답
        self.yname = yname                          # 꽃이름을 가지고 있는 리스트
        self.dim = xtrain[0].__len__()              # dimension 즉, data set feature 종류
        self.disarr = []                            # distance array 줄임말, [거리, 답] 을 담고있는 리스트
        self.vote_cnt = [[0., 0], [0., 1], [0., 2]] # vote count 줄임말, [가중치, 답] 을 담고있는 리스트 0,1,2 가 얼마나 나왔느냐
        self.grapcor = []                           # 그래프 출력을 하기위해 mv, wmv 의 결과를 순서대로 저장하는 전역 리스트

    def show_dim(self): # 변수 dim 을 출력하는 함수
        print(self.dim)

    def get_nearest_k(self, xtest_i):     # 가장가까운 점부터 순서대로 disarr 리스트에 저장해주는 함수
        for j in range(len(self.xtrain)): # for 문 (train data set 모두에 대하여)
            res = 0.                      # disarr 리스트에 담을 거리 값
            for i in range(0, self.dim):  # dimension 즉, feature 갯수만큼에 대하여 거리를 계산
                res = res + (xtest_i[i] - self.xtrain[j][i])**2 # (train0 - test0)^2 + ... + (train3 - test3)^2
            res = res**0.5                                      # 마지막에 0.5 승으로 루트를 씌워준다.
            self.disarr.append((res, int(self.ytrain[j])))      # disarr에 [res, 답] 을 append 해준다.
        self.disarr.sort(key=itemgetter(0))         # 최종적으로 disarr를 res 기준으로 sorting 한다 (오름차순)

    def mv(self):   # Majority Vote 함수
        for j in range(0, self.k):                          # for 문 (k 값 만큼)
            self.vote_cnt[self.disarr[j][1]][0] += 1        # 가중치는 그냥 갯수이므로 +1 로 해주고, disarr의 0부터 k개를 본다.
        self.vote_cnt.sort(key=itemgetter(0), reverse=True) # vote_cnt에서 가장큰 가중치를 알기 위해 sorting reverse (내림차순)
        self.grapcor.append(self.vote_cnt[0][1])            # 결과를 grapcor 리스트에 넣어준다.
        return self.yname[self.vote_cnt[0][1]]              # 결과를 꽃이름으로 반환한다.

    def wmv(self):  # Weighted Majority Vote 함수
        for j in range(0, self.k):                          # for 문 (k 값 만큼)
            self.vote_cnt[self.disarr[j][1]][0] += (1 / self.disarr[j][0]) # 가중치는 1/거리, disarr의 0부터 k개를 본다.
        self.vote_cnt.sort(key=itemgetter(0), reverse=True) # vote_cnt에서 가장큰 가중치를 알기 위해 sorting reverse (내림차순)
        self.grapcor.append(self.vote_cnt[0][1])            # 결과를 grapcor 리스트에 넣어준다.
        return self.yname[self.vote_cnt[0][1]]              # 결과를 꽃이름으로 반환한다.

    def set_k(self, k):      # k 값을 설정하는 set 함수
        self.k = k

    def get_grapcor(self):   # grapcor 리스트를 반환해주는 get 함수
        return self.grapcor

    def reset_grapcor(self): # grapcor 리스트를 초기화하는 함수
        self.grapcor = []

    def reset(self):         # disarr 리스트와 vote_cnt 리스트를 초기화 하는 함수
        self.disarr = []
        self.vote_cnt = [[0., 0], [0., 1], [0., 2]]