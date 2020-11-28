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



class lda_clustering:

  
    DEFAULT_PREPROCESSING_OUTPUT = "/dbfs/FileStore/vector_no_stopw_df.parquet"

    """Optimal Parameters Obtained for lda
    """
    DEFAULT_OPTIMAL_NUMBER = 150
    MAXITER = 100


    DEFAULT_OUTPUT_FILE = "../data/LDA.csv"
    sqlContext = SQLContext(sc)


    
    def lda_optimal(self, preprocess_file = DEFAULT_PREPROCESSING_OUTPUT, optimal_K = DEFAULT_OPTIMAL_NUMBER, maxiter = MAXITER, output_file_name = DEFAULT_OUTPUT_FILE):

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
		# TF
		cv = CountVectorizer(inputCol="vector_removed", outputCol="raw_features", vocabSize=5000, minDF=5.0)
		cvmodel = cv.fit(df_txts)
		result_cv = cvmodel.transform(df_txts)
		# IDF
		idf = IDF(inputCol="raw_features", outputCol="features")
		idfModel = idf.fit(result_cv)
		result_tfidf = idfModel.transform(result_cv)

		lda = LDA(k=150, maxIter=100)
		ldaModel = lda.fit(result_tfidf)
		output = ldaModel.transform(result_tfidf)
		find_cluster = udf(lambda row: int(np.argmax(row)))

		temp = output.withColumn('cluster_id',find_cluster(col("topicDistribution")))
		temp.select("id","cluster_id", "features").toPandas().to_csv(output_file_name, index = False, header = True)



    def lda_tuning(self, k_range, preprocess_file = DEFAULT_PREPROCESSING_OUTPUT):
        
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
		# TF
		cv = CountVectorizer(inputCol="vector_removed", outputCol="raw_features", vocabSize=5000, minDF=5.0)
		cvmodel = cv.fit(df_txts)
		result_cv = cvmodel.transform(df_txts)


        perplexity = 100000
        optimal_K = 0
        for k in k_range:
            try:
                lda = LDA(k=k, maxIter=100)
				ldaModel = lda.fit(result_tfidf)
				lp = ldaModel.logPerplexity(result_tfidf)
                if lp > perplexity:
                    optimal_K = k
            except:
                pass
        
        return optimal_K



if __name__ == "__main__":

    algo = lda_clustering() 
	algo.lda_optimal(preprocess_file = "/dbfs/FileStore/vector_no_stopw_df.parquet", output_file_name = "../data/LDA.csv") 


