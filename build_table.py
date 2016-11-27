import argparse
import json
import re
from collections import defaultdict

if __name__ == '__main__':
    items_filename = 'items_all.json'
    manual_filename = 'manual_test.json'
    min_freq = 0.001
    max_freq = 0.09
    load_manual = True
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
        test_label = labels
        test_data = wikiDesc
    
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
    valid_words = {k:v for k,v in word_count.iteritems()\
                   if (v >= min_freq*num_doc) and (v <= max_freq*num_doc)}
    print len(word_count), len(valid_words)