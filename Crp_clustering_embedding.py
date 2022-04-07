
### This is a simple Implementation of clustering with No limitaion of
### detection the Number of cluster like K-meanse. We Use CRP Algorithm
### to Cluster Data(tweets) as you can See the cycle of Algorithm in this link below.
### https://www.statisticshowto.com/chinese-restaurant-process/

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
print("the end of down and import")
import os
import pandas as pd
#########################################
import re
import preprocessor as p
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import bz2, random, numpy
#from wordcloud import WordCloud
print("End of importing")
################---Loading DataSet---#################
df = pd.read_csv('Data_Tweets/train_E6oV3lV.csv')

################Function to find cosine similarity###################
def cosinesim(v1, v2):
    return numpy.dot(v1, v2)/(numpy.linalg.norm(v1)* numpy.linalg.norm(v2))

########### custum function to clean the dataset (combining tweet_preprocessor and reguar expression) ########
def clean_tweets(df):
    #set up punctuations we want to be replaced
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
    REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")
    tempArr = []
    for line in df:
        # send to tweet_processor
        tmpL = p.clean(line)
        # remove puctuation
        tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower()) # convert all tweets to lower cases
        tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
        tempArr.append(tmpL)
    return tempArr

# Call Function to clean the tweets
df['clean tweet'] = clean_tweets(df['tweet'])
print(df['label'].head(20))

corpus = list(df['clean tweet'])
####### Here we Only Use first 50 of dataset, you can use all Dataset #######
corpus = corpus[0:50]

###### Use of Related madule and function of sentence embedding  #######
corpus_embeddings = embedder.encode(corpus)

print("shape of embedding vector :: ",corpus_embeddings.shape)

#############################################
############Start Of clustering CRP #########
#corpus_embeddings.tolist()

clusterVec = []  # tracks sum of vectors in a cluster


vecs = corpus_embeddings.tolist()
clusterIdx = [[]]  # array of index arrays. e.g. [[1, 3, 5], [2, 4, 6]]
ncluster = 0
# probablity to create a new table if new customer
#If It's not strongly "similar" to any existing table
pnew = 1.0 / (1 + ncluster)
N = len(vecs)

#rands = [random.random() for x in range(N)]  # N rand variables sampled from U(0, 1)
###################################
for k in range(3):
    print("*")
df_cluster = pd.DataFrame()
###################################
cols=['label']
lst=[]
for i in range(N):
    maxSim = -1
    maxIdx = 0
    v = vecs[i]
    #j = 0
    for j in range(ncluster):


        sim =  cosinesim(v,vecs[j])
        if sim > maxSim:
            maxIdx = j
            maxSim = sim
            # probablity to create a new table if new customer
            # is not strongly "similar" to any existing table
            if maxSim < pnew and j==ncluster-1 :
                #if (rands[i] < pnew and j==ncluster-1):
                    clusterVec.append(v)
                    clusterIdx.append([i])
                    #df_cluster['label'] = df_cluster.append({'label': i}, ignore_index=True)
                    lst.append([maxIdx+1])
                    df1 = pd.DataFrame(lst, columns=cols)

                    ncluster += 1
                    pnew = 1.0 / (1 + ncluster)
                    aaa = 1.0 / (1 + 1)
                    continue
        if (j==ncluster-1):
             clusterIdx[maxIdx] = clusterIdx[maxIdx] + [i]
             #df_cluster['label'] = df_cluster.append({'label': i}, ignore_index=True)
             lst.append([maxIdx])
             df1 = pd.DataFrame(lst, columns=cols)

    if (i==0):
        #clusterIdx[maxIdx] = clusterIdx[maxIdx] + [i]
        clusterVec.append(v)
        clusterIdx[maxIdx] = clusterIdx[maxIdx] + [i]
        lst.append([ncluster])
        df1 = pd.DataFrame(lst, columns=cols)

    if (ncluster == 0):
        ncluster += 1
#### In ourput the index of Data Id in its own cluster will be shown
print("End of process")
print("clusterIdx ::" , clusterIdx) ## List of last Clusters
print("clusterIdx ::" , len(clusterIdx))
print("df1 :: ",df1.head(20))

######################################
