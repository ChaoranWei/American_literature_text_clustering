import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer

stopwords = nltk.corpus.stopwords.words("english")
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)   
    
vocab_frame = pd.DataFrame({'words': totalvoab_tokenized}, index = totalvocab_stemmed)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorzer = TfidfVectorizer(max_df = 0.8, max_features = 200000, min_df = 0.2, 
                                  stopwords = 'english',use_idf = True, 
                                  tokenizer = tokenize_and_stem, ngram_range = (1,3))
    

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)
print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

from sklearn.metric.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters = num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

from sklearn.externals import joblib

joblib.dump(km, 'cluster.pkl')

km = joblib.load('cluster.pkl')
clusters = km.labels_.tolist()

films = { 'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters,
          'genre': genres }

frame = pd.DataFrame(films, index = [clusters], columns = ['rank', 'title', 'cluster', 'genre'])
frame['cluster'].value_counts()

grouped = frame['rank'].groupby(frame['cluster'])
grouped.mean()

from __future__ import print_function

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:,::-1]

for i in range(num_clusters):
    print("cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0]
              .encode('utf-8', 'ignore'), end = ',')
    print()
    print()
    
    print("Cluster %d titles:" % i, end = '')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end = '')
        
    print()
    print()
    
    
#dimensionality reduction

import os

import matplotlib.pyplot as plt
import matploblib as mpl

from sklearn.manifold import MDS

MDS()

mds = MDS(n_components = 2, dissimilarity = "precomputed")
pos = mds.fit_transform(dist)
xs, ys = pos[:, 0], pos[:, 1]

print()
print()

cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

cluster_names = {0: 'Family, home, war', 
                 1: 'Police, killed, murders', 
                 2: 'Father, New York, brothers', 
                 3: 'Dance, singing, love', 
                 4: 'Killed, soldiers, captain'}

df = pd.DataFrame(dict(x=xs, y=ys, label = clusters, title = titles))
groups = df.groupby('label')

fig, ax = plt.subplot(figsize = (17, 9))
ax.margins(0.05)

for name, group in groups:
    ax.plot(group.x, group.y, marker = 'o', linestyle='', ms = 12,
            label=cluster_names[name], color=cluster_colors[name],
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
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  
    
plt.show()
#plt.savefig('clusters_small_noaxes.png', dpi=200)

plt.close()

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