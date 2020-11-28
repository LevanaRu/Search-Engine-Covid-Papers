import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import GaussianMixture


class gmm():

    optimal_k = 150
    DEFAULT_PREPROCESSING_OUTPUT = "../preprocessing/preprocessing_output_0_7.csv"
    DEFAULT_OUTPUT_FILE = "gmm_output.csv"

    def clustering_optimal(self):
        df_raw = pd.read_csv(f"{self.DEFAULT_PREPROCESSING_OUTPUT}", header=None)

        spark = SparkSession \
            .builder \
            .appName("PySparkKMeans") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()

        df = spark.createDataFrame(df_raw)
        assembler = VectorAssembler(inputCols=df.columns, outputCol="features")
        # df_sample = df.sample(withReplacement=False, fraction=0.1)
        df_vec = assembler.transform(df).select("features")



        # gmm
        gm = GaussianMixture(k=self.optimal_k, tol=1, seed=10)
        gm.setMaxIter(500)

        model = gm.fit(df_vec)
        model.setPredictionCol("newPrediction")
        transformed = model.transform(df).select("features", "newPrediction")

        transformed = transformed.reset_index()
        transformed.to_csv(f"{self.DEFAULT_OUTPUT_FILE}")
        return transformed

    def clustering_tuning(self):
        df_raw = pd.read_csv(f"{self.DEFAULT_PREPROCESSING_OUTPUT}", header=None)

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

        transformed = transformed.reset_index()
        return transformed
