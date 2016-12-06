#!/usr/bin/env python

import argparse
import json
import re
import math
from collections import defaultdict
from operator import mul
import gzip


if __name__ == '__main__':
    items_filename = 'items_all.json'
    manual_filename = 'manual_test.json'
    wiki_filename = 'wikidev.json'

    #min_freq = 0.001
    #max_freq = 0.09
    min_num = 3
    max_num = 2000
    with open(items_filename,mode='rt') as fp:
        items = json.load(fp)
    data = []
    # extract descriptions
    for item in items:
        name, fb, mayo, wiki = item['fb']['name'], item['fb'], item['mayo'], item['wiki']
        wikiDesc = '\n'.join(wiki['mainSectionText'])
        if wikiDesc == None: wikiDesc = ""
        data.append((name.lower(), '\n'.join(fb['desc']), '\n'.join(mayo['definition']), wikiDesc))
    # extract symptoms
    for item in items:
        name, fb, mayo, wiki = item['fb']['name'], item['fb'], item['mayo'], item['wiki']
        def getTextFromWikiItem(item):
            text = ""
            for section in [(item['articleTitle'], item['mainSectionText'])] + item['subSections']:
                text = text + section[0] + '\n'
                for paragraph in section[1]:
                    text = text + paragraph + '\n'
            return text
        data.append((name.lower(), '\n'.join(fb['sym']), '\n'.join(mayo['symptoms']), getTextFromWikiItem(wiki)))

    # extract independent data
    labels = [x[0] for x in data]
    fbV = [x[1] for x in data]
    mayoV = [x[2] for x in data]
    wikiV = [x[3] for x in data]

    # create boring data
    train_label = 2*labels
    train_data = fbV + mayoV# + wikiV

    #build tokenizer
    class Tokenizer:
        def __init__(self):
            self.pattern = re.compile(r'(?u)\b\w\w+\b')
            self.filter = re.compile('[^0-9][a-zA-z]+[^0-9]')
            #self.filter = re.compile('.*[a-zA-z]+.*')
            with open('stop.json', 'rt') as fp:
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
            # remove stop
            res = [x for x in res if x not in self.stop]
            # stem
            res2 = []
            for x in res:
                if len(x) > 3 and x[-3:] == 'ies' and (x[-4] not in ['a','e']):
                    x = x[:-3]
                elif len(x) > 2 and x[-2:] == 'es' and (x[-3] not in ['a','e','o']):
                    x = x[:-2]
                elif len(x) > 1 and x[-1:] == 's' and (x[-2] not in ['u','s']):
                    x = x[:-1]
                if len(x) >= 2 and x[-2:] == 'ly':
                    x = x[:-2]
                if len(x) >= 3 and x[-3:] == 'ing':
                    x = x[:-3]
                elif len(x) >= 2 and x[-2:] == 'ed':
                    x = x[:-2]
                if len(x) > 0 and x[-1] == 'e':
                    x = x[:-1]
                if len(x) > 0 and x[-1] == 'y':
                    x = x[:-1]
                if len(x) > 0 and x[-1] == 'i':
                    x = x[:-1]
                if len(x) > 0:
                    res2.append(x)
            res = res2

            # remove stop
            res = [x for x in res if x not in self.stop]

            return res

    tkn = Tokenizer()
    #get words that appear in enough and not too many documents
    word_count = defaultdict(float)
    num_doc = len(train_data)
    for doc in fbV + mayoV:
        for x in set(tkn.run(doc)):
            word_count[x] += 1.0
    valid = {k:v for k,v in word_count.items()\
                   if (v >= min_num) and (v <= max_num)}
    #               if (v >= min_freq*num_doc) and (v <= max_freq*num_doc)}
    valid_words = list(valid.keys())
    #build idf
    idf = {k:math.log((1.0+num_doc)/v+1.0)+1.0 for k,v in valid.items()}
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

    print(len(word_count), len(valid_words))

    # learn a classifier
    n = len(valid_words)
    alpha = 1.0
    prob = defaultdict(lambda: defaultdict(float))
    for lbl,tfi in zip(train_label,tf):
        vec = {k: v*idf[k] for k,v in tfi.items()}
        for word,v in vec.items():
            prob[lbl][word] += v
    total_counts = {k:sum(v.values()) for k,v in prob.items()}
    normz = {c: math.log(1.0/float(total_counts[c]+alpha*n)) for c in total_counts}
    alphas = {}
    for c,v in prob.items():
        for word,weight in v.items():
            prob[c][word] = math.log(weight+alpha)
    theta = prob
    def returnScores(text,tkn,model,idf,alphas):
        tokens = [x for x in tkn.run(text) if x in valid_words]
        tfi = defaultdict(float)
        for token in tokens:
            tfi[token] += 1.0
        for token in tokens:
            tfi[token] = math.log(tfi[token]) + 1.0
        vec = {k: v*idf[k] for k,v in tfi.items()}

        scores = {}
        for c in model:
            scores[c] = sum([vec[k]*(model[c][k]+normz[c]) for k in vec])
        maxs = max(scores.values())
        expsum = sum([math.exp(x-maxs) for x in list(scores.values())])
        total_score = math.log(expsum) if expsum != 0  else 0.0
        scores = sorted([(math.exp(v-total_score-maxs),k) for k,v in scores.items()])[::-1]
        return scores
    for load_manual in [True,False]:
        if load_manual:
            with open(manual_filename,'rt') as fp:
                md = json.load(fp)
            test_label = []
            test_data = []
            for name in md:
                test_label.append(name.lower())
                test_data.append(md[name])
        else:
            with open(wiki_filename,'rt') as fp:
                wd = json.load(fp)
            test_label = []
            test_data = []
            for data,name in wd:
                test_label.append(name.lower())
                test_data.append(data)
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
        dataset = 'manual' if load_manual else 'wiki'
        print("{2}\t{0:.2f}\t{1:.2f}".format(float(top_1)/total,float(top_5)/total,dataset))
        for c in theta:
            bad_w = []
            for w in theta[c]:
                if theta[c][w] == 0.0:
                    bad_w.append(w)
            for w in bad_w:
                theta[c].pop(w, None)

    with gzip.open('data_table.json.gz','wt') as fp:
        json.dump({'idf':idf,'prob':theta,'norm':normz},fp,sort_keys=True,indent=4, separators=(',', ': '))
