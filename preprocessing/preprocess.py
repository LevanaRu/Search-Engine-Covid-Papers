import numpy as np 
import pandas as pd
import glob
import json

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from IPython.utils import io
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_sci_lg  # model downloaded in previous step

import string
import time


class Preprocessing:

    DEFAULT_OUTPUT_FILE = "data/preprocessing_output.csv"
    DEFAULT_ROOT_PATH = "data/downloaded_file"

    def preprocess(self, raw_data = DEFAULT_ROOT_PATH, output_file = DEFAULT_OUTPUT_FILE):
        metadata_path = f'{raw_data}/metadata.csv'
        meta_df = pd.read_csv(metadata_path, dtype={
            'pubmed_id': str,
            'Microsoft Academic Paper ID': str, 
            'doi': str
        })
        #print(meta_df.head())

        all_json = glob.glob(f'{raw_data}/pdf_json/**/*.json', recursive=True)
        #print(len(all_json))

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

        first_row = FileReader(all_json[0])

        def get_breaks(content, length):
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


        dict_ = {'paper_id': [], 'doi':[], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}
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
                summary = get_breaks(' '.join(info), 40)
                dict_['abstract_summary'].append(summary + "...")
            else:
                # abstract is short enough
                summary = get_breaks(content.abstract, 40)
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

        df_covid = pd.DataFrame(dict_, columns=['paper_id', 'doi', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])
        df_covid.head()
        #df_covid.to_csv("/data/jilin/4225proj/df_covid.csv", index = False)


        df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))  # word count in abstract
        df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))  # word count in body
        df_covid['body_unique_words']=df_covid['body_text'].apply(lambda x:len(set(str(x).split())))  # number of unique words in body
        df_covid.head()

        #df_covid['abstract'].describe(include='all')

        df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)
        #df_covid['abstract'].describe(include='all')

        #df_covid['body_text'].describe(include='all')

        #print(df_covid.describe())

        df = df_covid
        df.dropna(inplace=True)

        from tqdm import tqdm
        from langdetect import detect
        from langdetect import DetectorFactory

        # set seed
        DetectorFactory.seed = 0

        # hold label - language
        languages = []

        # go through each text
        for ii in tqdm(range(0,len(df))):
            # split by space into list, take the first x intex, join with space
            text = df.iloc[ii]['body_text'].split(" ")

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
                        lang = detect(df.iloc[ii]['abstract_summary'])
                    except Exception as e:
                        lang = "unknown"
                        pass

            # get the language
            languages.append(lang)

        from pprint import pprint

        languages_dict = {}
        for lang in set(languages):
            languages_dict[lang] = languages.count(lang)
            
        #print("Total: {}\n".format(len(languages)))
        pprint(languages_dict)

        df['language'] = languages
        df = df[df['language'] == 'en']
        #print(df.info())

        import string

        punctuations = string.punctuation
        stopwords = list(STOP_WORDS)
        stopwords[:10]

        custom_stop_words = [
            'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
            'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
            'al.', 'Elsevier', 'PMC', 'CZI', 'www'
        ]

        for w in custom_stop_words:
            if w not in stopwords:
                stopwords.append(w)

        # Parser
        parser = en_core_sci_lg.load(disable=["tagger", "ner"])
        parser.max_length = 7000000

        def spacy_tokenizer(sentence):
            mytokens = parser(sentence)
            mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
            mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
            mytokens = " ".join([i for i in mytokens])
            return mytokens

        tqdm.pandas()
        df["processed_text"] = df["body_text"].progress_apply(spacy_tokenizer)
        #df['body_word_count'].describe()
        #df['body_unique_words'].describe()

        from sklearn.feature_extraction.text import TfidfVectorizer
        def vectorize(text, maxx_features):

            vectorizer = TfidfVectorizer(max_features=maxx_features)
            X = vectorizer.fit_transform(text)
            return X

        text = df['processed_text'].values
        X = vectorize(text, 2 ** 12)
        X.shape

        from sklearn.decomposition import PCA

        pca = PCA(n_components=0.8, random_state=42)
        X_reduced = pca.fit_transform(X.toarray())
        #print(X_reduced.shape)

        np.savetxt(output_file, X_reduced, delimiter = ",")


if __name__ == "__main__":
    
    prep = Preprocessing()
    prep.preprocess()
