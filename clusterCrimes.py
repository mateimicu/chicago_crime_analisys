from pyspark.sql import SparkSession
import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
from matplotlib import pyplot as plt
from matplotlib import cm
import sys


def drawClusters(prediction, model, k, name):
    color_list = [cm.jet(x) for x in np.linspace(0.0, 1.0, k) ]
    plt.clf()
    plt.figure(figsize=(13,8))  
    for i in range(0, k):
        lx, ly, lcluster = [], [], []
        cluster_pre = prediction.select("features", "prediction").where(prediction.prediction==i).limit(200)
        for x, y in cluster_pre.collect():
            lx.append(x[0])
            ly.append(x[1])
        plt.scatter(lx, ly, c=color_list[i], marker="x", label="cluster " + str(i + 1))
    

    cx, cy, cc, count = [], [], [], 0
    for x, y in model.clusterCenters():
        cx.append(x)
        cy.append(y)
        cc.append(color_list[count])
        count += 1    
    plt.scatter(cx, cy, s=500, facecolors='none', edgecolors='black', marker="o",linewidths=3)
    plt.scatter(cx, cy, s=500, c=cc, marker="*")
    plt.title('2D Chart')
    plt.legend(loc='lower right')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.grid()
    plt.savefig(name + ".pdf")


sparkConf = SparkSession.builder.appName("Cluster Crimes").getOrCreate()
data = sparkConf.read.format("csv").option("header", "true").option("delimiter", ",").option("inferschema", "true").load(sys.argv[1])
data = data.cache()

data = data.withColumnRenamed('X Coordinate', 'xcoord')
data = data.withColumnRenamed('Y Coordinate', 'ycoord')
data = data.where(data.Latitude.isNotNull() & data.Longitude.isNotNull())
data = data.where(data['xcoord'] != 0.0)
data = data.where(data['ycoord'] != 0.0)
data = data.cache()

df = data['xcoord', 'ycoord'].rdd.map(lambda x: (Vectors.dense([float(x.xcoord), float(x.ycoord)]), )).toDF(["features"])

trainingData, testData = df.randomSplit([0.8, 0.2])

kmeansScores = []
for k in range(4, 8):
    kmeans = KMeans().setK(k).setSeed(216)
    model = kmeans.fit(trainingData)
    prediction = model.transform(testData)
    evaluator = ClusteringEvaluator()
    score = evaluator.evaluate(prediction)
    kmeansScores.append(score)

plt.plot(range(4, 8), kmeansScores, 'ro')
plt.savefig('kmeansScores.pdf')

bisectScores = []
for k in range(4, 8):
    bisection = BisectingKMeans().setK(k).setSeed(216)
    model = bisection.fit(trainingData)
    prediction = model.transform(testData)
    evaluator = ClusteringEvaluator()
    score = evaluator.evaluate(prediction)
    bisectScores.append(score)

plt.plot(range(4, 8), bisectScores, 'g^')
plt.savefig('bisectScores.pdf')
plt.clf()

kmeansK = np.argmax(kmeansScores) + 4
bisectK = np.argmax(bisectScores) + 4

evaluator = ClusteringEvaluator()
kmeans = KMeans().setK(kmeansK).setSeed(216)
kmModel = kmeans.fit(trainingData)
kmPrediction = kmModel.transform(testData)
bisection = BisectingKMeans().setK(bisectK).setSeed(216)
biModel = bisection.fit(trainingData)
biPrediction = biModel.transform(testData)

kMeansPredictionCount = kmPrediction.groupBy('prediction').count().sort('prediction')
bisectPredictionCount = biPrediction.groupBy('prediction').count().sort('prediction')

kMeansPredictionCount.show()
bisectPredictionCount.show()

drawClusters(kmPrediction, kmModel, kmeansK, "kmeansClusters")
drawClusters(biPrediction, biModel, bisectK, "bisectionClusters")