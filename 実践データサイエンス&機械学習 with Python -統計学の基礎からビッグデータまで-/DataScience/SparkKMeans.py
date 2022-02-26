# -*- coding: utf-8 -*-
from pyspark.mllib.clustering import KMeans
from numpy import array, random
from math import sqrt
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import scale

K = 5

# Sparkの設定
conf = SparkConf().setMaster("local").setAppName("SparkKMeans")
sc = SparkContext(conf = conf)

# N人をK個にクラスター化した、収入と年齢のフェイクのデータを作成する
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range (k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])
    X = array(X)
    return X

# データを読み込む。scale関数でデータの正規化を行う。
data = sc.parallelize(scale(createClusteredData(100, K)))

# モデルの訓練を行う（クラスタリングする）
clusters = KMeans.train(data, K, maxIterations=10,
        runs=10, initializationMode="random")

# クラスターの割り当てをprintする
resultRDD = data.map(lambda point: clusters.predict(point)).cache()

print "Counts by value:"
counts = resultRDD.countByValue()
print counts

print "Cluster assignments:"
results = resultRDD.collect()
print results


# 誤差の二乗の合計の平方根による、クラスタリングの評価
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# 課題
# Kを増加させたり減少させたりすると、WSSSEにどのような変化が生じるでしょうか？
# クラスタリングの前にデータの正規化を行わないと、何が起きるでしょうか？
# maxIterationsやrunsのパラメータを変更すると、何が起きるでしょうか？
