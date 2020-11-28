from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark Bisecting Kmeans") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

print(spark.version)

# Actual data
import pandas as pd
df_raw = pd.read_csv("/dbfs/FileStore/tables/data_preprocess/part1.csv", header=None)
for file_num in range(2, 6):
#   print(file_num)
  file_data = pd.read_csv("/dbfs/FileStore/tables/data_preprocess/part{}.csv".format(file_num), header=None)
  df_raw = pd.concat([df_raw, file_data])
df_raw.head(5)

df = spark.createDataFrame(df_raw)
# df.show()
assembler = VectorAssembler(inputCols=df.columns,outputCol="features")
dataset=assembler.transform(df)
# dataset.select("features").show(truncate=False)

# Trains a bisecting k-means model.
bkm = BisectingKMeans().setK(50).setSeed(1)
model = bkm.fit(dataset)

# Make predictions
predictions = model.transform(dataset)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
print("Cluster Centers: ")
centers = model.clusterCenters()
count = 1
for center in centers:
  print("center", count, ":")
  print(center)
  count = count + 1
  