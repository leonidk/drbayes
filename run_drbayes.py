#!/usr/bin/env python

import argparse
import json
import re
import math
from collections import defaultdict
from operator import mul
import gzip
import codecs


if __name__ == '__main__':
    #build tokenizer
    class Tokenizer:
        def __init__(self):
            self.pattern = re.compile(r'(?u)\b\w\w+\b')
            self.filter = re.compile('.*[a-zA-z]+.*')

        def run(self,text):
            # extract 
            res = self.pattern.findall(text.lower())
            # remove alphanumeric
            res = [x for x in res if self.filter.match(x) != None]
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
            return res

    tkn = Tokenizer()
    with gzip.open('data_table.json.gz',mode='rt') as fp:
        data = json.load(fp)
    idf = data['idf']
    model = data['prob']
    norm = data['norm']
    valid_words = list(idf.keys())

    def returnScores(text,tkn,model,idf,norm):
        tokens = [x for x in tkn.run(text) if x in valid_words]
        tfi = defaultdict(float)
        for token in tokens:
            tfi[token] += 1.0
        for token in tokens:
            tfi[token] = math.log(tfi[token]) + 1.0
        vec = {k: v*idf[k] for k,v in tfi.items()}

        scores = {}
        for c in model:
            rs = 0.0
            for k in vec:
                ms = model[c][k] if k in model[c] else 0.0
                rs += vec[k] * (ms + norm[c])
            scores[c] = rs
        maxs = max(scores.values())
        expsum = sum([math.exp(x-maxs) for x in list(scores.values())])
        total_score = math.log(expsum) if expsum != 0  else 0.0
        scores = sorted([(math.exp(v-total_score-maxs),k) for k,v in scores.items()])[::-1]
        return scores
    
    while True:
        try: input = raw_input
        except NameError: pass
        test_string = input("What's your problem?  ")
        scores = returnScores(test_string,tkn,model,idf,norm)
        print("Top outputs")
        for prob, c in scores[:5]:
            print('{0:.2f}, {1}'.format(prob,c))
        print('')
