from operator import itemgetter # 리스트 sorting 시 특정 key 값으로 정렬하기 위함

class KNN():
    def __init__(self, k, xtrain, ytrain, yname):
        self.k = k                                  # KNN 에서의 k 값
        self.xtrain = xtrain                        # train data set feature
        self.ytrain = ytrain                        # train data set 답
        self.yname = yname                          # 숫자 0~9 이름 리스트
        self.dim = xtrain[0].__len__()              # dimension 즉, data set feature 종류
        self.ylen = yname.__len__()                 # yname 길이
        self.disarr = []                            # distance array 줄임말, [거리, 답] 을 담고있는 리스트
        self.vote_cnt = [[0., 0], [0., 1], [0., 2], [0., 3] ,[0., 4], [0., 5], [0., 6], [0., 7], [0., 8], [0., 9]]
        self.grapcor = []                           # 그래프 출력을 하기위해 mv, wmv 의 결과를 순서대로 저장하는 전역 리스트
        self.sqrtdim_double = int((xtrain[0].__len__() ** 0.5)*2)   # 784개 feature을 (28+28)개의 feature로 계산한값
        self.hand_xtrain = []                                       # (56+56)개의 feature로 변환한 train data를 저장하는 리스트

    def show_dim(self): # 변수 dim 을 출력하는 함수
        print("dim : ", self.dim)

    def change_feature(self, data_i):             # 데이터 하나를 넣으면 784개 feature을 (56 + 56)개의 feature로 변환하는 함수
        dim_len = int(self.sqrtdim_double / 2)      # sqrtdim_double = 56이다. dim_len = 28
        flist = []                                  # (56 + 56)개의 feature로 변환된 데이터를 담을 리스트
        # 행에 대하여
        for i in range(0, dim_len):                 # 28 행
            (x_max, x_min, res) = (-1, dim_len, 0)  # 색칠되어있는 최대 index, 최소 index, 색칠된 갯수 res 초기값 설정
            for j in range(0, dim_len):             # 28 열
                if data_i[i*dim_len + j] > (5/255): # train데이터의 [i][j]값이 5/255 이상이면 칠해져있다고 판단
                    res += 1                        # 색칠된 수 +1
                    if x_max < j:                   # 최대 index가 현재 index보다 작다면
                        x_max = j                   # 최대 index는 현재 index로
                    if x_min > j:                   # 최소 index가 현재 index보다 크다면
                        x_min = j                   # 최소 index는 현재 index로
            if x_max == -1 or x_min == dim_len:     # 만일 아무것도 칠해지지 않은 경우이다
                flist.append(0)                         # i 행 검은색 사이 빈 공간 수(0) feature로 append
            else:                                   # 칠해져 있는게 있는 경우
                flist.append(x_max - x_min - res + 1)   # i 행 검은색 사이 빈 공간 수 feature로 append
            flist.append(res)                       # i 행에 칠해져 있는 갯수 feature로 append
        # 열에 대하여
        for i in range(0, dim_len):                 # 28 행
            (y_max, y_min, res) = (-1, dim_len, 0)  # 색칠되어있는 최대 index, 최소 index, 색칠된 갯수 res 초기값 설정
            for j in range(0, dim_len):             # 28 열
                if data_i[j*dim_len + i] > (5/255): # train데이터의 [j][i]값이 5/255 이상이면 칠해져있다고 판단 열을
                    res += 1                        # 색칠된 수 +1
                    if y_max < i:                   # 최대 index가 현재 index보다 작다면
                        y_max = i                   # 최대 index는 현재 index로
                    if y_min > i:                   # 최소 index가 현재 index보다 크다면
                        y_min = i                   # 최소 index는 현재 index로
            if y_max == -1 or y_min == dim_len:     # 만일 아무것도 칠해지지 않은 경우이다
                flist.append(0)                         # i 열 검은색 사이 빈 공간 수(0) feature로 append
            else:                                   # 칠해져 있는게 있는 경우
                flist.append(y_max - y_min - res + 1)   # i 열 검은색 사이 빈 공간 수 feature로 append
            flist.append(res)                       # i 열에 칠해져 있는 갯수 feature로 append
        return flist                                # flist 반환

    def handcraft_feature(self):                # 기존 train 데이터의 feature를 변환한 데이터를 hand_xtrain 리스트에 채워주는 함수
        for k in range(len(self.xtrain)):       # trian 데이터의 크기 만큼 for 문
            self.hand_xtrain.append(self.change_feature(self.xtrain[k])) # change_feature함수를 적용, hand_xtrain에 append

    def get_nearest_k(self, xtest_i, option):     # 가장가까운 점부터 순서대로 disarr 리스트에 저장해주는 함수
        if option == 'feature_compression':       # option이 'feature_compression' 인 경우
            dim_len = self.sqrtdim_double * 2     # dim_len = 56 * 2 = 112
            xtesti = self.change_feature(xtest_i) # test데이터의 feature를 변경한다.
            x_train = self.hand_xtrain            # 변경된 feature를 가진 train데이터를 사용한다.
        else :
            dim_len = self.dim                    # feature수 = self.dim
            xtesti = xtest_i                      # test데이터 그대로 사용
            x_train = self.xtrain                 # train데이터 그대로 사용

        for j in range(len(x_train)): # for 문 (train data set 모두에 대하여)
            res = 0.                           # disarr 리스트에 담을 거리 값
            for i in range(0, dim_len):        # dimension 즉, feature 갯수만큼에 대하여 거리를 계산
                res = res + (float(xtesti[i]) - float(x_train[j][i]))**2 # (train0 - test0)^2 + ... + (trainN - testN)^2
            res = res**0.5                                          # 마지막에 0.5 승으로 루트를 씌워준다.
            self.disarr.append((res, int(self.ytrain[j])))          # disarr에 [res, 답] 을 append 해준다.
        self.disarr.sort(key=itemgetter(0))                         # 최종적으로 disarr를 res 기준으로 sorting 한다 (오름차순)

    def mv(self):   # Majority Vote 함수
        for j in range(0, self.k):                          # for 문 (k 값 만큼)
            self.vote_cnt[self.disarr[j][1]][0] += 1        # 가중치는 그냥 갯수이므로 +1 로 해주고, disarr의 0부터 k개를 본다.
        self.vote_cnt.sort(key=itemgetter(0), reverse=True) # vote_cnt에서 가장큰 가중치를 알기 위해 sorting reverse (내림차순)
        self.grapcor.append(self.vote_cnt[0][1])            # 결과를 grapcor 리스트에 넣어준다.
        return self.yname[self.vote_cnt[0][1]]              # 결과를 이름으로 반환한다.
    def wmv(self):  # Weighted Majority Vote 함수
        for j in range(0, self.k):                          # for 문 (k 값 만큼)
            self.vote_cnt[self.disarr[j][1]][0] += (1 / self.disarr[j][0]) # 가중치는 1/거리, disarr의 0부터 k개를 본다.
        self.vote_cnt.sort(key=itemgetter(0), reverse=True) # vote_cnt에서 가장큰 가중치를 알기 위해 sorting reverse (내림차순)
        self.grapcor.append(self.vote_cnt[0][1])            # 결과를 grapcor 리스트에 넣어준다.
        return self.yname[self.vote_cnt[0][1]]              # 결과를 이름으로 반환한다.

    def set_k(self, k):      # k 값을 설정하는 set 함수
        self.k = k
    def get_grapcor(self):   # grapcor 리스트를 반환해주는 get 함수
        return self.grapcor
    def rest_votecnt(self):
        self.vote_cnt = [[0., 0], [0., 1], [0., 2], [0., 3] ,[0., 4], [0., 5], [0., 6], [0., 7], [0., 8], [0., 9]]
    def reset(self):         # disarr 리스트와 vote_cnt 리스트를 초기화 하는 함수
        self.disarr = []
        self.vote_cnt = [[0., 0], [0., 1], [0., 2], [0., 3] ,[0., 4], [0., 5], [0., 6], [0., 7], [0., 8], [0., 9]]