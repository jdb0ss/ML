from collections import Counter
from cassiopeia.data import Season, Queue
from cassiopeia import Summoner, Match
import cassiopeia as cass
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

cass.set_riot_api_key("RGAPI-c4c7f4de-ef97-4904-ad40-ff8ba5a147e6")
cass.set_default_region("KR")
summonerName = "hide on bush"

# xy = np.loadttxt('test.csv', delimiter=',', dtype=np.float32)

summoner = cass.get_summoner(name=summonerName)

match_history = summoner.match_history
match_history(seasons={Season.season_9}, queues={Queue.ranked_solo_fives})

x_train = []
y_train = []

for i in range(20, 40):
    for p in match_history[i].participants:
        if p.summoner.id == summoner.id:
            x_train.append(p.stats.kda)
            y_train.append(p.stats.win)

print("Train Data Set")
for i in range(0, 20):
    print("kda : " + str(x_train[i]) + ", win : " + str(y_train[i]))


W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = x_train * W + b
cost = tf.reduce_mean(tf.square(hypothesis-y_train))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # Fit the line
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])

        if step % 100 == 0:
            print(step, cost_val, W_val, b_val)

print(W_val[0], b_val[0])

P_X = []
P_Y = []

for i in range(0, 20):
    for p in match_history[i].participants:
        if p.summoner.id == summoner.id:
            P_X.append(W_val[0]*p.stats.kda+b_val[0])
            P_Y.append('승' if p.stats.win else '패')

for i in range(0, 20):
    print("예상 승률 : {0}% 실제 결과 : {1}".format(round(P_X[i]*100, 2), P_Y[i]))
