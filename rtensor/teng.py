import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()
import cassiopeia as cass
from cassiopeia import Summoner, Match
from cassiopeia.data import Season, Queue

flg = plt.figure()
ax = flg.gca(projection ='3d')

cass.set_riot_api_key("RGAPI-91b72378-8538-4678-8249-e5d07974c0f0")
cass.set_default_region("KR")
summonerName = "hide on bush"

trainNum = 100
summoner = cass.get_summoner(name=summonerName)
match_history = summoner.match_history
match_history(seasons={Season.season_9}, queues={Queue.ranked_solo_fives})

global_glod = [0,0]
gold_diff = []
kda = []
x1_train = []
x2_train = []
x3_train = []
y_train = []
X_train = []

for i in range(testNum,trainNum+testNum):
    flag = 0
    global_glod[0] = global_glod[1] = 0
    for p in match_history[i].red_team.participants:
        if p.summoner.id == summoner.id : flag = 0
        global_glod[0]+=p.stats.gold_earned
    for p in match_history[i].blue_team.participants:
        if p.summoner.id == summoner.id : flag = 1
        global_glod[1]+=p.stats.gold_earned
    for p in match_history[i].participants:
        if p.summoner.id == summoner.id:
            kda.append(p.stats.kda)
            gold_diff.append(global_glod[flag] - global_glod[1-flag])
            x3_train.append(p.team.first_baron)
            y_train.append([p.stats.win])

print("Train Data Set")
for i in range(0,trainNum):
    print("kda : {0}, gold_diff : {1}$, first_baron : {2}, win : {3}"
    .format(round(kda[i],2),gold_diff[i],x3_train[i],'승' if y_train[i] else '패'))
for i in range(0,trainNum):
    normalized = (kda[i] - np.mean(kda)/np.std(kda))
    x1_train.append(normalized)
    normalized = (gold_diff[i] - np.mean(gold_diff))/np.std(gold_diff)
    x2_train.append(normalized)
    X_train.append([x1_train[i],x2_train[i],x3_train[i]])

X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 1])
W = tf.Variable(tf.random_normal([3,1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(50000):
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict = {X : X_train, Y : y_train})

        if step % 10000 == 0:
            print(step, "Cost: {0}\n, w : {1}, b : {2}".format(cost_val,W_val,b_val))

ret_gold_diff = []
ret_kda = []
x1_ret = []
x2_ret = []
x3_ret = []
y_ret = []
P_X = []
P_Y = []
X_test = []

for i in range(0,testNum):
    flag = 0
    global_glod[0] = global_glod[1] = 0
    for p in match_history[i].red_team.participants:
        if p.summoner.id == summoner.id : flag = 0
        global_glod[0]+=p.stats.gold_earned
    for p in match_history[i].blue_team.participants:
        if p.summoner.id == summoner.id : flag = 1
        global_glod[1]+=p.stats.gold_earned
    for p in match_history[i].participants:
        if p.summoner.id == summoner.id:
            ret_kda.append(p.stats.kda)
            ret_gold_diff.append(global_glod[flag] - global_glod[1-flag])
            x3_ret.append(p.team.first_baron)
            y_ret.append(p.stats.win)

for i in range(0,testNum):
    normalized = (ret_kda[i] - np.mean(ret_kda))/np.std(ret_kda)
    x1_ret.append(normalized)
    normalized = (ret_gold_diff[i] - np.mean(ret_gold_diff))/np.std(ret_gold_diff)
    x2_ret.append(normalized)
    X_test.append([x1_ret[i],x2_ret[i],x3_ret[i]])

hypothesis = tf.sigmoid(tf.matmul(X_test, W_val) + b_val)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    P_X = sess.run(hypothesis)
    for i in range(0,testNum):
        P_Y.append('승' if y_ret[i] else '패')
print(P_X)
rate = 0
for i in range(0,testNum):
    print("예상 승률: {0}%, 실제 결과 : {1}".format(round(P_X[i][0]*100,2),P_Y[i]))
    if P_X[i][0]>=0.5 and P_Y[i]=='승' : rate = rate+1
    if P_X[i][0]<0.5 and P_Y[i]=='패' : rate = rate+1
print("예측 성공률 : {0}%".format(round(rate*100/testNum,2)))

ax.scatter(x1_ret,x2_ret,x3_ret,c=y_ret,cmap='Greys')
ax.set_xlabel('kda')
ax.set_ylabel('gold_diff')
ax.set_zlabel('first_baron')

plt.show()