import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import GaussianMixture


class gmm():
    root_path = "/dbfs/FileStore/tables/data_preprocess/"
    optimal_k = 150

    def clustering(self):
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

        K_lst = list(range(50, 401, 10))

        # gmm
        for k in range(K_lst):
            gm = GaussianMixture(k=k, tol=1, seed=10)
            gm.setMaxIter(500)

            model = gm.fit(df_vec)
            model.setPredictionCol("newPrediction")
            transformed = model.transform(df).select("features", "newPrediction")

        return transformed
