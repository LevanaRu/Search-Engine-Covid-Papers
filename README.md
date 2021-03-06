# Search-Engine-Covid-Papers
Creating a search engine for covid-19 papers.

Due to the file size restriction of LumiNUS, only a subset of data is put in this folder for a demonstration purpose.


## Dependencies:
Spark version: 3.0.1  
Scala version: 2.12  
Pyspark version 3.0.1  
BERT set-up instruction: https://bert-as-service.readthedocs.io/en/latest/  
sparkml-som set-up instruction: https://github.com/FlorentF9/sparkml-som  
The specific execution instruction is written in the main function of each script. 

## 1. Data Pre-processing
The script will pre-process the input json file into a clean text format and vectorised formate after dimension reduction. 
It will be  passed to clustering algorithm in the next step. 

## 2. Clustering
The various clustering algorithm uses high dimentional vector input and output a dataframe, containing the document id and its corresponding cluster number. 

## 3. Topic Modelling
Topic modeling takes in the clustered documents and analyse the keyword in each cluster. Top M important keywords are taken to be the tagging for each topic. Relevant documents for each topic can also be retrieved.

## 4. Searching
Searching makes use of the keyword tagging and search for the most similar cluster to the input search query. 
