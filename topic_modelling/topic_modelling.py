from pyspark.sql import SparkSession, Row
import pandas as pd
import pyspark
from pyspark.sql import SQLContext

from pyspark.ml.feature import CountVectorizer , IDF
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, IDF, RegexTokenizer, StopWordsRemover
from pyspark.ml.clustering import LDA, LocalLDAModel
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import explode
from pyspark.sql.functions import flatten
import pyspark.sql.types as T

from pyspark.sql.types import *
from pyspark.sql.functions import col



class topic_modelling:

  
    DEFAULT_PREPROCESSING_OUTPUT = "/dbfs/FileStore/vector_no_stopw_df.parquet"
    CLUSTER_DF = "../data/cluster.csv"

    """Optimal Parameters Obtained for lda
    """
    DEFAULT_OPTIMAL_NUMBER = 150
    MAXITER = 100


    DEFAULT_OUTPUT_FILE = "../data/topic_modelling.csv"
    sqlContext = SQLContext(sc)

    
    def lda_optimal(self, preprocess_file = DEFAULT_PREPROCESSING_OUTPUT, cluster_df = CLUSTER_DF, maxiter = MAXITER, 
    	output_file_name = DEFAULT_OUTPUT_FILE, max_term_tagging = m):

		def is_digit(value):
	    if value:
	        return value.isdigit()
	    else:
	        return False
		filter_number_udf = udf(lambda row: [x for x in row if not is_digit(x)], ArrayType(StringType()))
		temp = qlContext.read.parquet(preprocess_file)
		temp = temp.withColumn('no_number_vector_removed', filter_number_udf(col('vector_no_stopw')))
		temp1= temp.select(temp.paper_id,explode(temp.no_number_vector_removed))
		temp2 = temp1.filter(temp1.col != "")
		temp3 = temp2.groupby("paper_id").agg(F.collect_list("col").alias("vector_removed"))
		inner_join = temp3.join(temp, ["paper_id"])
		windowSpec  = Window.orderBy(F.col("paper_id"))
		df_final = inner_join.withColumn("id",F.row_number().over(windowSpec))
		df_txts = df_final.select("vector_removed", "id","paper_id", "doi", "title", "authors", "abstract", "abstract_summary", "vector_no_stopw")
		df = sqlContext.read.format("com.databricks.spark.csv")
    					.option("header", "true")
    					.option("inferschema", "true")
    					.option("mode", "DROPMALFORMED")
    					.load("CLUSTER_DF")
    	df_txts = df.join(tedf_txtsmp, "paper_id" = "index")

		# TF
		cv = CountVectorizer(inputCol="vector_removed", outputCol="raw_features", vocabSize=5000, minDF=5.0)
		cvmodel = cv.fit(df_txts)
		result_cv = cvmodel.transform(df_txts)
		# IDF
		idf = IDF(inputCol="raw_features", outputCol="features")
		idfModel = idf.fit(result_cv)
		result_tfidf = idfModel.transform(result_cv)

		from pyspark.sql import SparkSession
		from pyspark.sql.types import StructType,StructField, StringType

		spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()

		schema = StructType([
		  StructField('cluster_id', StringType(), True),
		  StructField('tagging', ArrayType(), True)
		  ])

		topic_modeling = spark.createDataFrame(spark.sparkContext.emptyRDD(),schema)
	
		distinct_clusters = result_tfidf.select("cluster_id").distinct().sorted().collect_list()
		for i in distinct_clusters: 
			subset = result_tfidf.filter(result_tfidf.cluster_id == i)
			lda = LDA(k=1, maxIter=100)
			ldaModel = lda.fit(result_subset)
			output = ldaModel.transform(result_tfidf)
			if (i == 0):
				full_df = output
			else:
				full_df = full_df.union(output)
			topics = ldaModel.describeTopics(maxTermsPerTopic=m)
			vocabArray = cvmodel.vocabulary
			ListOfIndexToWords = udf(lambda wl: list([vocabArray[w] for w in wl]))
			FormatNumbers = udf(lambda nl: ["{:1.4f}".format(x) for x in nl])

			taggings = topics.select(ListOfIndexToWords(topics.termIndices).alias('words'))
			temp = spark.createDataFrame(
							    [(i, taggings)],
							    ['cluster_id', 'taggings'] 
							)
			topic_modeling = topic_modeling.union(temp)

		# output the taggings of each topic
		topic_modeling.to_csv(output_file_name)

		return full_df




	def find_relevant(cluster_id, df, documents_to_display):
		df_sliced = df.select("id","topicDistribution","title", "abstract_summary").rdd.map(lambda r: Row(ID=int(r[0]), weight=float(r[1][ntopic]), title = r[2], abstract_summary = r[3])).toDF()
	    df_need = df_sliced.sort(df_sliced.weight.desc())
	    print("Relevant documents are: ")
	    for i in range(documents_to_display):
	      DocIDs = df_need.take(cluster_id)
	      print(DocIDs[i])
	    print('===================================================')

    


if __name__ == "__main__":

    algo = topic_modelling() 
	df = algo.lda_optimal(preprocess_file = "/dbfs/FileStore/vector_no_stopw_df.parquet", output_file_name = "../data/LDA.csv") 



