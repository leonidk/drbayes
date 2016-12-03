import argparse
import json
import re
import math
from collections import defaultdict
from operator import mul
import gzip

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
            return res

    tkn = Tokenizer()
    with gzip.open('data_table.json.gz') as fp:
        data = json.load(fp)
    idf = data['idf']
    model = data['prob']
    norm = data['norm']
    valid_words = idf.keys()

    def returnScores(text,tkn,model,idf,norm):
        tokens = [x for x in tkn.run(text) if x in valid_words]
        tfi = defaultdict(float)
        for token in tokens:
            tfi[token] += 1.0
        for token in tokens:
            tfi[token] = math.log(tfi[token]) + 1.0
        vec = {k: v*idf[k] for k,v in tfi.iteritems()}

        scores = {}
        for c in model:
            rs = 0.0
            for k in vec:
                ms = model[c][k] if k in model[c] else 0.0
                rs += vec[k] * (ms + norm[c])
            scores[c] = rs
        maxs = max(scores.values())
        expsum = sum([math.exp(x-maxs) for x in scores.values()])
        total_score = math.log(expsum) if expsum != 0  else 0.0
        scores = sorted([(math.exp(v-total_score-maxs),k) for k,v in scores.iteritems()])[::-1]
        return scores
    
    while True:
        input = raw_input("What's your problem?  ")
        scores = returnScores(input,tkn,model,idf,norm)
        print "Top outputs"
        for prob, c in scores[:5]:
            print '{0:.2f}, {1}'.format(prob,c)
        print ''
