# =============================================================================
# ################################## imports ##################################
# =============================================================================

import pandas as pd
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from sklearn.cluster import KMeans
import re
import nltk
import string
import re


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nrclex import NRCLex
import matplotlib.pyplot as plt
import text2emotion as te


# =============================================================================
# ################################# get data ##################################
# =============================================================================

allLyrics = pd.read_csv('taylorswiftdata')

# =============================================================================
# ############################### preprocessing ###############################
# =============================================================================

#add genre labels
country = ['Taylor Swift', 'Fearless', 'Speak Now', 'Red']
pop = ['1989', 'Reputation', 'Lover', 'Midnights']
alternative = ['folklore', 'evermore']

genres = []

for album in allLyrics['Album']:
    if album in country:
        genres.append('Country') 
    if album in pop:
        genres.append('Pop') 
    if album in alternative:
        genres.append('Alternative')
        

allLyrics['Genre'] = genres

#make lowercase
allLyrics['Lyrics'] = allLyrics['Lyrics'].apply(lambda txt: txt.lower())

#remove non-letters
allLyrics['Lyrics'] = allLyrics['Lyrics'].str.replace('[^\w\s]','')
allLyrics['Lyrics'] = allLyrics['Lyrics'].str.replace(r'[0-9]', '')

#split
allLyrics['Lyrics'] = allLyrics['Lyrics'].apply(lambda txt: txt.split())

#remove stop words
def removeStop(txt):
    stop_words = set(stopwords.words('english'))
    words = []
    for t in txt:
        if t not in stop_words:
            words.append(t)
    
    return words

allLyrics['Lyrics'] = allLyrics['Lyrics'].apply(lambda txt: removeStop(txt))

# =============================================================================
# ################################# clustering ################################
# =============================================================================  

#create embeddings
lyrics = allLyrics['Lyrics'].tolist()

#embed = Word2Vec(lyrics, vector_size = 100, min_count = 1)

embed = api.load('word2vec-google-news-300')

vec = embed.wv

vec = KeyedVectors.load_word2vec_format(embed, binary=True, limit=500000)


#test embeddings (lowkey doesn't make sense...)
embed.most_similar('love')
embed.most_similar('hate')

#avg vectors for each song
lyricvec = []

for song in lyrics:
    vecs = [embed[word] for word in song if word in embed]
    if vecs:
        lyricvec.append(sum(vecs)/len(vecs))
        
vecmatrix = np.array(lyricvec)

#elbow plot to find number of clusters (best = 4)
wcss = []

for i in range(1,15):
    kmeans = KMeans(n_clusters = i,
                    init = 'k-means++',
                    max_iter = 300,
                    n_init = 10,
                    random_state = 0)
    kmeans.fit(vecmatrix)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 15), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
    
#create clusters
kmeans = KMeans(n_clusters = 4, 
                init='k-means++', 
                max_iter = 300,
                n_init = 10, 
                random_state = 0)

#pull cluster labels
labels = kmeans.fit_predict(vecmatrix)

#plot
plt.scatter(vecmatrix[:, 0], vecmatrix[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#add cluster labels to main df
allLyrics['label'] = labels

#group by album
albumGroup = allLyrics.groupby(['label', 'Album']).size().unstack(fill_value=0)

#group by genre
genreGroup = allLyrics.groupby(['label', 'Genre']).size().unstack(fill_value=0)

#group by clusters
cluster1 = (allLyrics.loc[(allLyrics.label == 0)]).drop(columns = 'Lyrics')
cluster2 = (allLyrics.loc[(allLyrics.label == 1)]).drop(columns = 'Lyrics')
cluster3 = (allLyrics.loc[(allLyrics.label == 2)]).drop(columns = 'Lyrics')
cluster4 = (allLyrics.loc[(allLyrics.label == 3)]).drop(columns = 'Lyrics')
cluster5 = (allLyrics.loc[(allLyrics.label == 4)]).drop(columns = 'Lyrics')
 
# =============================================================================
# ################################# sentiment #################################
# ============================================================================= 

#join text
allLyrics['Lyrics'] = allLyrics['Lyrics'].apply(lambda txt: ' '.join(txt))

#create sentiment (pos/neg/neutral) function
sentiment = SentimentIntensityAnalyzer()

def sent(txt):
    score = sentiment.polarity_scores(txt)
    sent = 1 if score['compound'] > 0 else 0
    return score

#get polarity score + emotion scores for each song
scores = []
emotions = []

test = allLyrics['Lyrics'][0]
print(te.get_emotion(test))

print(NRCLex(test).raw_emotion_scores)


for lyric in allLyrics['Lyrics']:
    pol = sent(lyric)
    scores.append(pol)
    
    emo = NRCLex(lyric).raw_emotion_scores
    
    emotions.append(emo)

#seperate emotion scores
allLyrics['Emotions'] = emotions

allEmotions = pd.json_normalize(allLyrics['Emotions'])

allLyrics = pd.concat([allLyrics, allEmotions],
                      axis = 1,
                      join = 'inner')


allLyrics = allLyrics.fillna(0)

allLyrics2 = allLyrics.drop(columns = 'Emotions')

