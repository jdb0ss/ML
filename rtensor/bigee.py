import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cassiopeia as cass
from cassiopeia.data import Season, Queue
cass.set_riot_api_key("RGAPI-91b72378-8538-4678-8249-e5d07974c0f0")
cass.set_default_region("KR")
summonerName = "hide on bush"
trainNum = 100
flg = plt.figure()
ax = flg.gca(projection ='3d')
summoner = cass.get_summoner(name=summonerName)
match_history = summoner.match_history
match_history(seasons={Season.season_9}, queues={Queue.ranked_solo_fives})

global_glod = [0,0]
gold_diff = []
kda = []
x1tmp_train = []
x1_train = []
x2_train = []
x3_train = []
X_train = []

for i in range(0, trainNum):
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
            if p.team.first_baron == True:
                x3_train.append(1)
            else:
                x3_train.append(0)

for i in range(0,trainNum):
    normalized = ((kda[i] - np.mean(kda))/np.std(kda))
    x1tmp_train.append(normalized)
for i in range(0,trainNum):
    normalized = ((x1tmp_train[i] - np.min(x1tmp_train))/(np.max(x1tmp_train) - np.min(x1tmp_train) ))
    x1_train.append(normalized)
    normalized = (gold_diff[i] - np.mean(gold_diff))/np.std(gold_diff)
    x2_train.append(normalized)
    X_train.append([x1_train[i], x2_train[i], x3_train[i]])

points_n = trainNum
clusters_n = 2
iteration_n = 200 #상수 값
points = tf.constant(X_train)
centroids = tf.constant(tf.slice(tf.compat.v1.random_shuffle(points), [0, 0], [clusters_n, -1]))
points_expanded = tf.expand_dims(points, 0)
@tf.function
def update_centroids(points_expanded, centroids):
    centroids_expanded = tf.expand_dims(centroids, 1)

    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
    assignments = tf.argmin(distances, 0)
    means = []
    for c in range(clusters_n):
        ruc = tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1])
        ruc = tf.gather(points, ruc)
        ruc = tf.reduce_mean(ruc, axis=[1])
        means.append(ruc)
    new_centroids = tf.concat(means, 0)
    return new_centroids, assignments

for _ in range(iteration_n):
    centroids, assignments = update_centroids(points_expanded, centroids)

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=assignments , alpha=0.5)
ax.set_xlabel('kda')
ax.set_ylabel('gold_diff')
ax.set_zlabel('first_baron')
plt.show()