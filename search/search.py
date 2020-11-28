from bert_serving.client import BertClient
import pandas as pd
import time 

from sklearn.metrics.pairwise import cosine_similarity


class searching:

    DEFAULT_TAGGING_TXT = "cluster_taggings.txt"
    DEFAULT_RETURN_SIZE = 5
    DEFAULT_OUTPUT_FILE = "search_output_index.txt"

    def find_nearest(self, search_query, return_size = DEFAULT_RETURN_SIZE, taggings = DEFAULT_TAGGING_TXT, output_file = DEFAULT_OUTPUT_FILE):
        # Get an input query
        client = BertClient() 
        search_query = str(search_query)
        v1 = client.encode([search_query])

        # Read in all cluster taggings
        f = open(taggings, "r")

        d = {}
        for line in f:
            try: 
                line = line.rstrip("\n")
                line = line.rstrip(" ")
                line_split = line.split(", ")
                index = line_split[0].split(" ")[0]
                start = len(index) + 1
                index = int(index)
                line_split[0] = line_split[0][start:]
                for tag in line_split:
                    encoding = client.encode([tag])
                    d[tag] = [index, encoding]
            except:
                pass
        f.close()

        # Calculate cosine similarity between input query and all taggings
        similarity = {} # tagging: score
        topn_score = [] # top n scores
        for key, value in d.items():
            score = cosine_similarity(d[key][1], v1)[0][0]
            similarity[key] = score
            if len(topn_score) < return_size or topn_score is None:
                topn_score.append(score)
            else:
                if self.if_larger(topn_score, score):
                    topn_score[0] = score
            topn_score.sort()

        result_clusters = {} 
        for key, value in similarity.items(): 
            if similarity[key] in topn_score:
                if d[key][0] not in result_clusters.keys():
                    result_clusters[d[key][0]] = 0
                if result_clusters[d[key][0]] < similarity[key]:
                    result_clusters[d[key][0]] = topn_score[topn_score.index(similarity[key])]

        output = open(output_file, "w")
        output.write("cluster number, probability\n")
        for i in result_clusters.keys():
            output.write("{},{}\n".format(i, result_clusters[i]))
        output.close()
        return result_clusters

    def if_larger(self, lst, new_element):
        for i in range(len(lst)):
            if new_element > lst[i]:
                return True
            else:
                return False

if __name__ == "__main__":
    # Get an input query
    search_fun = searching()
    print(search_fun.find_nearest("covid"))
