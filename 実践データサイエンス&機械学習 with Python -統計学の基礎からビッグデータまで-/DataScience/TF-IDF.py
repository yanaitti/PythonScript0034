# -*- coding: utf-8 -*-
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

# Sparkの設定
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

# 文章の読み込み
rawData = sc.textFile("C:/DataScience/DataScience/subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))

# 文章の名前を取り出す
documentNames = fields.map(lambda x: x[1])

# TFの計算。各文章の単語はハッシュ値として数値化される。
hashingTF = HashingTF(100000) # ハッシュ値の数の上限
tf = hashingTF.transform(documents)

# IDFの計算
tf.cache()
idf = IDF(minDocFreq=2).fit(tf)

# 各文章における各単語のTF*IDFを計算する
tfidf = idf.transform(tf)

# 既にRDDをsparseベクトル（https://spark.apache.org/docs/latest/mllib-data-types.html）として得ています。
# 各TFxIDFの値は、各文章ごとにハッシュ値と関連付けて格納されています。

# 今回の元のデータに"Abraham Lincoln"の記事が含まれているので、 "Gettysburg"（リンカーンが有名なスピーチを行った場所）の単語で検索を行ってみましょう。

# "Gettysburg"のハッシュ値を取得します。
gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])

# Gettsyburgのハッシュ値に対応するTF*IDFのスコアを取り出し、各文章ごとにRDDに格納します。
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])

# 文章の名前とハッシュ値の関連付けを行います。
zippedResults = gettysburgRelevance.zip(documentNames)
print zippedResults.min()

# 最後に、最大のTF*IDFを持つ文章の名前をプリントします。 
print "Best document for Gettysburg is:"
print zippedResults.max()
