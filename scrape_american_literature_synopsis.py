import numpy as np
import requests
from pattern import web
import pickle
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def get_xml():
    url = 'http://americanliterature.com/books'
    return requests.get(url).text

def scrap_literature_data(xml):
    dom = web.Element(xml)
    result = {}
    span = dom.by_tag('span')[4]
    names = [n.content for n in span.by_tag('cite')]
    
    urls = [n.attributes['href'] for n in span.by_tag('a')]
    return names, urls

def scrap_book(names, urls):
    contents = {}
    for i, url in enumerate(urls):
        url = 'http://americanliterature.com/' + url
        dom = web.Element(requests.get(url).text)
        body = dom.by_tag('body')[0]
        suburls = [n.attributes['href'] for n in body.by_tag('a') if '/author/' in n.attributes['href']]
        contents[names[i]] = suburls
        print(names[i])
    return contents

def getText(Dict):
    newDict = {}
    for key, val in Dict.iteritems():
        temp = []
        for url in val:
            url = 'http://americanliterature.com/' + url
            dom = web.Element(requests.get(url).text)
            body = dom.by_tag('body')[0]
            suburls = [n.content for n in body.by_tag('p')]
            suburls = ' '.join(suburls)
            temp.append(suburls)
        temp = ' '.join(temp)
        if key + '.pkl' not in os.listdir('pickle'):
            pickle.dump(temp, open('pickle/' + key + '.pkl','wb'))
        print(key)
    return newDict
        
        
    
#names, urls = scrap_literature_data(get_xml())
#suburls = scrap_book(names, urls)
#pickle.dump(suburls, open('literature.pkl','wb'))
suburls = pickle.load(open('literature.pkl', 'rb'))
print('dictionary loaded')
text = getText(suburls)