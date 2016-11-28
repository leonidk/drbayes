import argparse
import json
import re
import math
from collections import defaultdict
from operator import mul

if __name__ == '__main__':
    items_filename = 'items_all.json'
    manual_filename = 'manual_test.json'
    wiki_filename = 'wikidev.json'

    min_freq = 0.001
    max_freq = 0.09
    load_manual = False
    with open(items_filename,'rb') as fp:
        items = json.load(fp)
    data = []
    # extract descriptions
    for item in items:
        name, fb, mayo, wiki = item['fb']['name'], item['fb'], item['mayo'], item['wiki']
        wikiDesc = '\n'.join(wiki['mainSectionText'])
        if wikiDesc == None: wikiDesc = ""
        data.append((name, '\n'.join(fb['desc']), '\n'.join(mayo['definition']), wikiDesc))
    # extract symptoms
    for item in items:
        name, fb, mayo, wiki = item['fb']['name'], item['fb'], item['mayo'], item['wiki']
        def getTextFromWikiItem(item):
            text = ""
            for section in [(item['articleTitle'], item['mainSectionText'])] + item['subSections']:
                text = text + section[0] + u'\n'
                for paragraph in section[1]:
                    text = text + paragraph + u'\n'
            return text
        data.append((name, '\n'.join(fb['sym']), '\n'.join(mayo['symptoms']), getTextFromWikiItem(wiki)))

    # extract independent data
    labels = [x[0] for x in data]
    fbV = [x[1] for x in data]
    mayoV = [x[2] for x in data]
    wikiDesc = [x[3] for x in data]

    # create boring data
    train_label = 2*labels
    train_data = fbV + mayoV

    if load_manual:
        with open(manual_filename,'rb') as fp:
            md = json.load(fp)
        test_label = []
        test_data = []
        for name in md:
            test_label.append(name)
            test_data.append(md[name])
    else:
        with open(wiki_filename,'rb') as fp:
            wd = json.load(fp)
        test_label = []
        test_data = []
        for data,name in wd:
            test_label.append(name)
            test_data.append(data)
        #test_label = labels
        #test_data = wikiDesc
    
    # get stop words

    #build tokenizer
    class Tokenizer:
        def __init__(self):
            self.pattern = re.compile(r'(?u)\b\w\w+\b')
            self.filter = re.compile('.*[a-zA-z]+.*')
            with open('stop.json', 'rb') as fp:
                d = json.load(fp)
                generalStopWords = d['gen']
                medicalStopWords = d['med']
                nltkStopWords = d['nltk']  
                unique = set()
                for word in generalStopWords:
                    unique.add(word.lower())
                for word in medicalStopWords:
                    unique.add(word.lower())
                for word in nltkStopWords:
                    unique.add(word.lower())
            self.stop = list(unique)
            self.vowels = re.compile(".*[aeiou].*")

        def run(self,text):
            # extract 
            res = self.pattern.findall(text.lower())
            # remove alphanumeric
            res = [x for x in res if self.filter.match(x) != None]
            # stem
            res2 = []
            for x in res:
                if x[-1] == 's': x = x[:-1]
                if x[-3:] == 'ing' and self.vowels.match(x[:-3]) != None : x = x[:-3]
                if x[-2:] == 'ed' and self.vowels.match(x[:-2]) != None : x = x[:-2]
                if x[-2:] == 'ly' and self.vowels.match(x[:-2]) != None : x = x[:-2]
                if x[-5:] == 'esses': x = x[:-5]
                if x[-3:] == 'ies': x = x[:-2]
                if x[-4:] == 'ness': x = x[:-4]
                if x[-3:] == 'ful': x = x[:-3]
                res2.append(x)
            res = res2
            
            # remove stop
            res = [x for x in res if x not in self.stop]
            return res

    tkn = Tokenizer()
    #get words that appear in enough and not too many documents
    word_count = defaultdict(float)
    num_doc = len(train_data)
    for doc in train_data:
        for x in set(tkn.run(doc)):
            word_count[x] += 1.0
    valid = {k:v for k,v in word_count.iteritems()\
                   if (v >= min_freq*num_doc) and (v <= max_freq*num_doc)}
    valid_words = valid.keys()
    #build idf
    idf = {k:math.log((1.0+num_doc)/v+1.0)+1.0 for k,v in valid.iteritems()}
    # build tf
    tf = []
    for doc in train_data:
        tokens = [x for x in tkn.run(doc) if x in valid_words]
        res = defaultdict(float)
        for token in tokens:
            res[token] += 1.0
        for token in tokens:
            res[token] = math.log(res[token]) + 1.0

        tf.append(res)
    
    print len(word_count), len(valid_words)

    # learn a classifier
    n = len(valid_words)
    alpha = 1.0
    prob = defaultdict(lambda: defaultdict(float))
    for lbl,tfi in zip(train_label,tf):
        vec = {k: v*idf[k] for k,v in tfi.iteritems()}
        for word,v in vec.iteritems():
            prob[lbl][word] += v
    total_counts = {k:sum(v.values()) for k,v in prob.iteritems()}
    alphas = {}
    for c,v in prob.iteritems():
        for word,weight in v.iteritems():
            prob[c][word] = (weight)/float(total_counts[c]+alpha*n)
            alphas[c] = alpha/float(total_counts[c]+alpha*n)
    #theta = {c: {word: math.log((weight+alpha)/(total_counts[c]+alpha*n))\
    #            for word,weight in vec.iteritems()} \
    #            for c,vec in prob.iteritems()}
    theta = prob
    def returnScores(text,tkn,model,idf,alphas):
        tokens = [x for x in tkn.run(text) if x in valid_words]
        tfi = defaultdict(float)
        for token in tokens:
            tfi[token] += 1.0
        for token in tokens:
            tfi[token] = math.log(tfi[token]) + 1.0
        vec = {k: v*idf[k] for k,v in tfi.iteritems()}

        scores = {}
        for c in model:
            scores[c] = sum(({k: v*(model[c][k]+alphas[c]) for k,v in vec.iteritems()}).values())
            #scores[c] = reduce(mul,(({k: v*(model[c][k]+alphas[c]) for k,v in vec.iteritems()}).values()))
        scores = sorted([(v,k) for k,v in scores.iteritems()])[::-1]
        return scores
    top_1 = 0
    top_5 = 0
    total = 0
    for lbl,data in zip(test_label,test_data):
        scores = returnScores(data,tkn,theta,idf,alphas)
        top5_lbl = [x[1] for x in scores[:5]]
        if lbl == top5_lbl[0]:
            top_1+=1
        if lbl in top5_lbl:
            top_5+=1
        total+=1
    print "{0:.2f}\t{1:.2f}".format(float(top_1)/total,float(top_5)/total)


