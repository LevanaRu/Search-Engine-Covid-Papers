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

class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
        def __repr__(self):
            return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'




class topic_modelling:

  
    DEFAULT_PREPROCESSING_OUTPUT = "/dbfs/FileStore/vector_no_stopw_df.parquet"
    CLUSTER_DF = "../data/dbscan_output.csv"

    """Optimal Parameters Obtained for lda
    """
    DEFAULT_OPTIMAL_NUMBER = 150
    MAXITER = 100


    DEFAULT_OUTPUT_FILE = "../data/topic_modelling.csv"
    sqlContext = SQLContext(sc)
	
    root_path = ".."
    metadata_path = "../data/metadata.csv"
    DEFAULT_INPUT_PATH = "../data/"
    DEFAULT_OUTPUT_FILE = "preprocessing_output_0_7.csv"

    def get_breaks(self, content, length):
        data = ""
        words = content.split(' ')
        total_chars = 0

        # add break every length characters
        for i in range(len(words)):
            total_chars += len(words[i])
            if total_chars > length:
                data = data + "<br>" + words[i]
                total_chars = 0
            else:
                data = data + " " + words[i]
        return data

    def clean_text(self,c):
        c = lower(c)
        c = regexp_replace(c, "^rt ", "")
        c = regexp_replace(c, "(https?\://)\S+", "")
        c = regexp_replace(c, "[^a-zA-Z0-9\\s]", "")
        # c = split(c, "\\s+") tokenization...
        return c

    def import_data(self):

        # meta df
        meta_df = pd.read_csv(self.metadata_path, dtype={
            'pubmed_id': str,
            'Microsoft Academic Paper ID': str,
            'doi': str
        })

        # json
        all_json = glob.glob(f"{self. DEFAULT_INPUT_PATH}/**/*.json", recursive=True)

        dict_ = {'paper_id': [], 'doi': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [],
                 'abstract_summary': []}
        for idx, entry in enumerate(all_json):
            if idx % (len(all_json) // 10) == 0:
                print(f'Processing index: {idx} of {len(all_json)}')

            try:
                content = FileReader(entry)
            except Exception as e:
                continue  # invalid paper format, skip

            # get metadata information
            meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
            # no metadata, skip this paper
            if len(meta_data) == 0:
                continue

            dict_['abstract'].append(content.abstract)
            dict_['paper_id'].append(content.paper_id)
            dict_['body_text'].append(content.body_text)

            # also create a column for the summary of abstract to be used in a plot
            if len(content.abstract) == 0:
                # no abstract provided
                dict_['abstract_summary'].append("Not provided.")
            elif len(content.abstract.split(' ')) > 100:
                # abstract provided is too long for plot, take first 100 words append with ...
                info = content.abstract.split(' ')[:100]
                summary = self.get_breaks(' '.join(info), 40)
                dict_['abstract_summary'].append(summary + "...")
            else:
                # abstract is short enough
                summary = self.get_breaks(content.abstract, 40)
                dict_['abstract_summary'].append(summary)

            # get metadata information
            meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

            try:
                # if more than one author
                authors = meta_data['authors'].values[0].split(';')
                if len(authors) > 2:
                    # if more than 2 authors, take them all with html tag breaks in between
                    dict_['authors'].append(get_breaks('. '.join(authors), 40))
                else:
                    # authors will fit in plot
                    dict_['authors'].append(". ".join(authors))
            except Exception as e:
                # if only one author - or Null valie
                dict_['authors'].append(meta_data['authors'].values[0])

            # add the title information, add breaks when needed
            try:
                title = get_breaks(meta_data['title'].values[0], 40)
                dict_['title'].append(title)
            # if title was not provided
            except Exception as e:
                dict_['title'].append(meta_data['title'].values[0])

            # add the journal information
            dict_['journal'].append(meta_data['journal'].values[0])

            # add doi
            dict_['doi'].append(meta_data['doi'].values[0])

        df_covid = pd.DataFrame(dict_,
                                columns=['paper_id', 'doi', 'abstract', 'body_text', 'authors', 'title', 'journal',
                                         'abstract_summary'])
        df_covid['abstract_word_count'] = df_covid['abstract'].apply(
            lambda x: len(x.strip().split()))  # word count in abstract
        df_covid['body_word_count'] = df_covid['body_text'].apply(
            lambda x: len(x.strip().split()))  # word count in body
        df_covid['body_unique_words'] = df_covid['body_text'].apply(
            lambda x: len(set(str(x).split())))  # number of unique words in body

        # remove duplicates
        df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)
        df_covid['abstract'].describe(include='all')
        df_covid.dropna(inplace=True)

        # handle multiple languages
        # set seed
        DetectorFactory.seed = 0

        # hold label - language
        languages = []

        # go through each text
        for ii in tqdm(range(0, len(df_covid))):
            # split by space into list, take the first x intex, join with space
            text = df_covid.iloc[ii]['body_text'].split(" ")

            lang = "en"
            try:
                if len(text) > 50:
                    lang = detect(" ".join(text[:50]))
                elif len(text) > 0:
                    lang = detect(" ".join(text[:len(text)]))
            # ught... beginning of the document was not in a good format
            except Exception as e:
                all_words = set(text)
                try:
                    lang = detect(" ".join(all_words))
                # what!! :( let's see if we can find any text in abstract...
                except Exception as e:

                    try:
                        # let's try to label it through the abstract then
                        lang = detect(df_covid.iloc[ii]['abstract_summary'])
                    except Exception as e:
                        lang = "unknown"
                        pass

            # get the language
            languages.append(lang)

        languages_dict = {}
        for lang in set(languages):
            languages_dict[lang] = languages.count(lang)

        df_covid['language'] = languages
        # drop
        df_covid = df_covid[df_covid['language']=='en']
        return df_covid

    def is_digit(self, value):
	if not value:
	    return False
	else: 
	    return value.isdigit()
	

    def lda_optimal(self, preprocess_file = DEFAULT_PREPROCESSING_OUTPUT, cluster_df = CLUSTER_DF, maxiter = MAXITER, output_file_name = DEFAULT_OUTPUT_FILE, max_term_tagging = m):

	filter_number_udf = udf(lambda row: [x for x in row if not is_digit(x)], ArrayType(StringType()))
	temp = sqlContext.read.parquet(preprocess_file)
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



