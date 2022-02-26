# -*- coding: utf-8 -*-
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array

# Sparkの設定
conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc = SparkContext(conf = conf)

# 数値データに変換
def binary(YN):
    if (YN == 'Y'):
        return 1
    else:
        return 0

def mapEducation(degree):
    if (degree == 'BS'):
        return 1       
    elif (degree =='MS'):
        return 2
    elif (degree == 'PhD'):
        return 3
    else:
        return 0

# MLLibのLabeledPointに変換
def createLabeledPoints(fields):
    yearsExperience = int(fields[0])
    employed = binary(fields[1])
    previousEmployers = int(fields[2])
    educationLevel = mapEducation(fields[3])
    topTier = binary(fields[4])
    interned = binary(fields[5])
    hired = binary(fields[6])

    return LabeledPoint(hired, array([yearsExperience, employed,
        previousEmployers, educationLevel, topTier, interned]))

# CSVファイルを読み込み、ヘッダーを取り除く 
rawData = sc.textFile("C:/DataScience/DataScience/PastHires.csv")
header = rawData.first()
rawData = rawData.filter(lambda x:x != header)

# 各行をカンマで分割し、LabeledPointsに変換
csvData = rawData.map(lambda x: x.split(","))
trainingData = csvData.map(createLabeledPoints)

# テスト用の候補者を作成。10年の経験、現在雇用されていて、3社を経験、学部卒、一流大学卒業ではない、インターンシップを経験していない。
testCandidates = [ array([10, 1, 3, 1, 0, 0])]
testData = sc.parallelize(testCandidates)

# 決定木による分類器を訓練する
model = DecisionTree.trainClassifier(trainingData, numClasses=2,
                                     categoricalFeaturesInfo={1:2, 3:4, 4:2, 5:2},
                                     impurity='gini', maxDepth=5, maxBins=32)

# 採用されるか、予測を行う
predictions = model.predict(testData)
print ('Hire prediction:')
results = predictions.collect()
for result in results:
    print result

# 決定木自体もprint可能
print('Learned classification tree model:')
print(model.toDebugString())
