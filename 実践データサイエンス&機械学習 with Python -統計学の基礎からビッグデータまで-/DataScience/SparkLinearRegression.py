# -*- coding: utf-8 -*-
from __future__ import print_function

from pyspark.ml.regression import LinearRegression

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

if __name__ == "__main__":

    # SparkSessionの作成 (注: configセクションはWindows用です)
    spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("LinearRegression").getOrCreate()

    # データを読み込み、MLLib用のフォーマットに変換する
    inputLines = spark.sparkContext.textFile("regression.txt")
    data = inputLines.map(lambda x: x.split(",")).map(lambda x: (float(x[0]), Vectors.dense(float(x[1]))))

    # RDDをDataFrameに変換する
    colNames = ["label", "features"]
    df = data.toDF(colNames)

    # データを訓練データとテストデータに分割する
    trainTest = df.randomSplit([0.5, 0.5])
    trainingDF = trainTest[0]
    testDF = trainTest[1]

    # 線形回帰のモデルを作成する
    lir = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # 訓練データを用いて、モデルを訓練する
    model = lir.fit(trainingDF)

    # テスト用のデータの特徴の値に線形回帰モデルを適用して、予測を行う
    fullPredictions = model.transform(testDF).cache()

    # 予測値を取り出す。そして、正解値を取り出してラベルとする。
    predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0])
    labels = fullPredictions.select("label").rdd.map(lambda x: x[0])

    # お互いを紐づける
    predictionAndLabel = predictions.zip(labels).collect()

    # 各点の予測値と実際の値をプリントする
    for prediction in predictionAndLabel:
      print(prediction)


    # セッションを停止する
    spark.stop()
