
import pandas as pd
import glob
import json
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from tqdm import tqdm
from langdetect import detect
from langdetect import DetectorFactory
from pprint import pprint
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import col, lower, regexp_replace, split
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_sci_lg  # model downloaded in previous step
import string
from pyspark.sql import functions as func
from pyspark.ml.feature import StopWordsRemover
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix





class preprocess_pyspark:
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
                    dict_['authors'].append(self.get_breaks('. '.join(authors), 40))
                else:
                    # authors will fit in plot
                    dict_['authors'].append(". ".join(authors))
            except Exception as e:
                # if only one author - or Null valie
                dict_['authors'].append(meta_data['authors'].values[0])

            # add the title information, add breaks when needed
            try:
                title = self.get_breaks(meta_data['title'].values[0], 40)
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


        # change to spark
        # Enable Arrow-based columnar data transfers
        spark = SparkSession \
            .builder \
            .appName("PySparkKMeans") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        spark.conf.set("spark.sql.execution.arrow.enabled", "true")

        # Create a Spark DataFrame from a pandas DataFrame using Arrow
        df_english = spark.createDataFrame(df_covid)
        clean_text_df = df_english.withColumn("text", self.clean_text(col("body_text")))

        tokenizer = Tokenizer(inputCol="text", outputCol="vector")
        vector_df = tokenizer.transform(clean_text_df)


        # remove stopwords
        punctuations = string.punctuation
        stopwords = list(STOP_WORDS)
        stopwords[:10]

        custom_stop_words = [
            'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
            'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
            'al.', 'elsevier', 'pmc', 'czi', 'www', "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
        ]

        for w in custom_stop_words:
            if w not in stopwords:
                stopwords.append(w)

        # Define a list of stop words or use default list
        remover = StopWordsRemover(stopWords=stopwords)

        # Specify input/output columns
        remover.setInputCol("vector")
        remover.setOutputCol("vector_no_stopw")

        # Transform existing dataframe with the StopWordsRemover
        vector_no_stopw_df = remover.transform(vector_df)



        # tdidf
        hashingTF = HashingTF()
        tf = hashingTF.transform(vector_no_stopw_df.select("vector_no_stopw"))

        # While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
        # First to compute the IDF vector and second to scale the term frequencies by IDF.
        tf.cache()
        idf = IDF().fit(tf)
        tfidf = idf.transform(tf)

        # PCA
        mat = RowMatrix(tfidf)
        # Compute the top 4 principal components.
        # Principal components are stored in a local dense matrix.
        pc = mat.computePrincipalComponents(1325)

        # Project the rows to the linear space spanned by the top 4 principal components.
        projected = mat.multiply(pc)
        projected.toPandas().to_csv(f"{self.DEFAULT_OUTPUT_FILE}")

        return projected


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


