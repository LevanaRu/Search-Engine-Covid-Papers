import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans
import pyspark.sql.functions as func


class kmeans_cluster():
    root_path = "/dbfs/FileStore/tables/data_preprocess/"
    optimal_k = 610

    def clustering_optimal(self):
        df_raw = pd.read_csv(f"{self.root_path}/part1.csv", header=None)
        for file_num in range(2, 7):
            #   print(file_num)
            file_data = pd.read_csv(f"{self.root_path}/part{file_num}.csv", header=None)
            df_raw = pd.concat([df_raw, file_data])

        spark = SparkSession \
            .builder \
            .appName("PySparkKMeans") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()

        df = spark.createDataFrame(df_raw)
        assembler = VectorAssembler(inputCols=df.columns, outputCol="features")
        # df_sample = df.sample(withReplacement=False, fraction=0.1)
        df_vec = assembler.transform(df).select("features")


        kmeans = KMeans(k=self.optimal_k)
        kmeans.setSeed(1)
        kmeans.setMaxIter(5000)

        model = kmeans.fit(df_vec)
        model.setPredictionCol("newPrediction")
        model.predict(df_vec.head().features)

        centers = model.clusterCenters()

        transformed = model.transform(df_vec).select("features", "newPrediction")
        rows = transformed.collect()

        # Evaluate clustering by computing Silhouette score
        evaluator = ClusteringEvaluator()
        transformed = transformed.withColumn("prediction", func.col("newPrediction"))
        transformed = transformed.reset_index()

        silhouette = evaluator.evaluate(transformed)

        return transformed

    def clustering_tuning(self):
        df_raw = pd.read_csv(f"{self.root_path}/part1.csv", header=None)
        for file_num in range(2, 7):
            #   print(file_num)
            file_data = pd.read_csv(f"{self.root_path}/part{file_num}.csv", header=None)
            df_raw = pd.concat([df_raw, file_data])

        spark = SparkSession \
            .builder \
            .appName("PySparkKMeans") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()

        df = spark.createDataFrame(df_raw)
        assembler = VectorAssembler(inputCols=df.columns, outputCol="features")
        # df_sample = df.sample(withReplacement=False, fraction=0.1)
        df_vec = assembler.transform(df).select("features")

        K_lst = list(range(100,10001,50))

        for k in K_lst:

            kmeans = KMeans(k=k)
            kmeans.setSeed(1)
            kmeans.setMaxIter(5000)

            model = kmeans.fit(df_vec)
            model.setPredictionCol("newPrediction")
            model.predict(df_vec.head().features)

            centers = model.clusterCenters()

            transformed = model.transform(df_vec).select("features", "newPrediction")
            rows = transformed.collect()

            # Evaluate clustering by computing Silhouette score
            evaluator = ClusteringEvaluator()
            transformed = transformed.withColumn("prediction", func.col("newPrediction"))
            transformed = transformed.reset_index()

            silhouette = evaluator.evaluate(transformed)

        return transformed


