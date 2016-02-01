# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
#import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer
import pickle
from sklearn.externals import joblib
stopwords = nltk.corpus.stopwords.words("english")
stemmer = SnowballStemmer("english")


textDict = {}
stopwords2 = ['\'s', 'n\’t',  '/b', 'b', '/i', 'br', 'href=', '/a','said', 'br br', 'look', 'like', 'did', 'time']

for i in os.listdir(os.getcwd() + '/pickle'):
    if i.endswith(".pkl"):
        print(i[:-4])
        
        text = pickle.load(open('pickle/'+ i, 'rb'))
        text = text.split(' ')
        text = [j for j in text if j != stopwords2]
        text = ' '.join(text)
        textDict[i[:-4]] = text


def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens
'''
print('tokenize and stem the documents...')

totalvocab_stemmed = []
totalvocab_tokenized = []
for i, j in textDict.iteritems():
    allwords_stemmed = tokenize_and_stem(j)
    totalvocab_stemmed.extend(allwords_stemmed)
    
    allwords_tokenized = tokenize_only(j)
    totalvocab_tokenized.extend(allwords_tokenized) 
    print('tokenize: ' +i)
'''
print('initialize vectorizer...')
    
#vocab_frame = pd.DataFrame({'words': totalvoab_tokenized}, index = totalvocab_stemmed)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df = 0.8, max_features = 20000, min_df = 0.05, 
                                  stop_words = 'english',use_idf = True, 
                                  tokenizer = tokenize_only, ngram_range = (1,2))
    
print('vectorizing...')
tfidf_matrix = tfidf_vectorizer.fit_transform(textDict.values())
print('tfidf matrix shape: ')
print(tfidf_matrix.shape)
print

terms = tfidf_vectorizer.get_feature_names()
print('first 100 terms: ')
print(terms[:100])
print

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
print('distance matrix: ')
print(dist)
print

joblib.dump(dist, 'dist.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(terms, 'terms.pkl')


from sklearn.cluster import KMeans

num_clusters = 10

km = KMeans(n_clusters = num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()
print('clusters:')
print(clusters)
print()

joblib.dump(km, 'cluster.pkl')

km = joblib.load('cluster.pkl')
terms = joblib.load('terms.pkl')
clusters = km.labels_.tolist()

lits = { 'title': textDict.keys(), 'cluster': clusters, 'text': textDict.values()}

#print lits['title']
#print lits['text'][0]

#frame = pd.DataFrame(films, index = [clusters], columns = ['rank', 'title', 'cluster', 'genre'])
#frame['cluster'].value_counts()

#grouped = frame['rank'].groupby(frame['cluster'])
#grouped.mean()


print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:,::-1] #coordinate of cluster center

num_clusters = 10
top_terms = {}
for i in range(num_clusters):
    print("cluster %d words:" % i, end='')
    temp = ' '
    for ind in order_centroids[i, :6]:
        print(' %s' % terms[ind]
              .encode('utf-8', 'ignore'), end = ',')
        temp += terms[ind] + ' '
    top_terms[i] = temp
    
    print()
    print()
    
    print("Cluster %d titles:" % i, end = '')
    cluster_title = [j for j, k in enumerate(lits['title']) if lits['cluster'][j] == i]
    for title in cluster_title:
        print(' %s,' % lits['title'][title], end = '')
        
    print()
    print()
    
 
#dimensionality reduction

import os

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

dist = joblib.load('dist.pkl')
MDS()

mds = MDS(n_components = 2, dissimilarity = "precomputed")
pos = mds.fit_transform(dist)
xs, ys = pos[:, 0], pos[:, 1]

print()
print()

cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}


#df = pd.DataFrame(dict(x=xs, y=ys, label = clusters, title = titles))
#groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(10, 6))
ax.margins(0.05)

for i in range(num_clusters):
    groupx = [xs[j] for j, k in enumerate(lits['cluster']) if lits['cluster'][j] == i]
    groupy = [ys[j] for j, k in enumerate(lits['cluster']) if lits['cluster'][j] == i]
    ax.plot(groupx, groupy, marker = 'o', linestyle='', ms = 12,
            label=top_terms[i], color=cluster_colors[i],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis = 'x',
                   which = 'both',
                   bottom='off',
                   top='off',
                   labelbottom='off')
    ax.tick_params(axis = 'y',
                   which = 'both',
                   left = 'off',
                   top = 'off',
                   labelleft='off')
    
ax.legend(numpoints = 1)

for i in range(len(lits['title'])):
    ax.text(xs[i], ys[i], lits['title'][i], size=8)  
    
plt.show()
plt.savefig('clusters.png', dpi=200)

plt.close()
'''
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}
        
#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

#group by cluster
groups = df.groupby('label')

#define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: -200px;}
"""

# Plot 
fig, ax = plt.subplots(figsize=(14,6)) #set plot size
ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, 
                     label=cluster_names[name], mec='none', 
                     color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]
    
    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())    
    
    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    
    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    
ax.legend(numpoints=1) #show legend with only one dot

mpld3.display() #show the plot

#uncomment the below to export to html
#html = mpld3.fig_to_html(fig)
#print(html)
'''