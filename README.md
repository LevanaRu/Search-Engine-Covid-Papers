# Search-Engine-Covid-Papers
Creating a search engine for covid-19 papers


## Dependencies:
Spark version: 3.0.1 
Scala version: 2.12
Pyspark version 3.0.1
The specific execution instruction is written in the main function of each script. 

## 1. Data Pre-processing
The script will pre-process the input json file into a clean text format and vectorised formate after dimension reduction. 
It will be  passed to clustering algorithm in the next step. 

## 2. Clustering
The various clustering algorith uses high dimentional vector input and output a dataframe， containing the document id and its corresponding cluster number. 

## 3. Topic Modelling
Topic modeling takes in the clustered documents and analyse the keyword in each cluster. Top M important keywords are taken to be the tagging for each topic. Relevant documents for each topic can also be retrievd.

## 4. Searching
Searching makes use of the keyword tagging and rsearch for the most similar cluster to the input search query. 
