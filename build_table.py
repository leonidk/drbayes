import argparse
import json

if __name__ == '__main__':
    items_filename = 'items_all.json'
    manual_filename = 'manual_test.json'
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
    
