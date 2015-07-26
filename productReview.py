
# coding: utf-8

# # Preparing training Data
# ####To be run only once

# In[1]:

import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import io
import re

def cleandata(mtext):
    mtext = re.sub(' +', ' ', mtext)
    mtext = re.sub('\n+', '.', mtext)
    #mtext = re.sub('[0-9]+. ', '', mtext)
    mtext = re.sub(' -', '.', mtext)
    #mtext = re.sub('\.\.+', '. ', mtext)
    mtext = re.sub('(?<=\\D)\.(?=\\D)', '. ', mtext)
    mtext = ''.join([i if ord(i) < 128 else ' ' for i in mtext])
    mtext = mtext.strip().lower()
    return mtext
    
trainprosraw = io.open("trainpwS.txt", "r", encoding='utf-8')
trainconsraw = io.open("traincwS.txt", "r", encoding='utf-8')

trainpros = []
traincons = []
for line in trainprosraw:
    trainpros.append(line[:-1])
for line in trainconsraw:
    traincons.append(line[:-1])


#trainpros = nltk.sent_tokenize(trainprosraw.read())
#traincons = nltk.sent_tokenize(trainconsraw.read())
#print trainpros
prowords = []
conwords = []
for sent in trainpros:
    prowords.extend(nltk.word_tokenize(sent))
for sent in traincons:
    conwords.extend(nltk.word_tokenize(sent))

trainprosraw.close()
trainconsraw.close()

stpwrds = stopwords.words()
#len(traincons)


# # Building the Classifier
# ####To be run only once

# In[2]:

def evaluate_classifier(featx):
    #negids = movie_reviews.fileids('neg')
    #posids = movie_reviews.fileids('pos')
    
    ##For Movie Review train:
    #negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    #posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
     
    ##For product reviews train:
    negfeats = [(featx([wrd for wrd in nltk.word_tokenize(con) if wrd not in stpwrds]), 'neg') for con in traincons]
    posfeats = [(featx([wrd for wrd in nltk.word_tokenize(pro) if wrd not in stpwrds]), 'pos') for pro in trainpros]
    
    negcutoff = len(negfeats)*3/4
    poscutoff = len(posfeats)*3/4
 
    trainfeats = negfeats[:] + posfeats[:]
    #trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
 
    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
 
    for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
 
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
    print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
    print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
    print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
    classifier.show_most_informative_features()
    return classifier
    
def word_feats(words):
    return dict([(word, True) for word in words])
 
#print 'evaluating single word features'
#evaluate_classifier(word_feats)
 
word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()
 
for word in prowords:
    word_fd[word.lower()]+=1
    label_word_fd['pos'][word.lower()]+=1
for word in conwords:
    word_fd[word.lower()]+=1
    label_word_fd['neg'][word.lower()]+=1

# n_ii = label_word_fd[label][word]
# n_ix = word_fd[word]
# n_xi = label_word_fd[label].N()
# n_xx = label_word_fd.N()
 
pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count
 
word_scores = {}
 
for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
        (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
        (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score

best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
bestwords = set([w for w, s in best])
 
def best_word_feats(words):
    return dict([(word, True) for word in words if word in bestwords])
 
#print 'evaluating best word features'
#mc=evaluate_classifier(best_word_feats)
 
def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d
 
print 'evaluating best words + bigram chi_sq word features'
mc = evaluate_classifier(best_bigram_word_feats)


# # Defining features DS

# In[16]:

features = [('Cpu/Processor', ''),
            ('Screen/Display', 'size/inch,resolution/ppi,type/ips/amoled'),
            ('Battery', 'standby,life/day/hour/hrs/usage'),
            ('Camera','front/selfie/secondary,rear/back/primary,flash,low light'),
            ('RAM/Memory', 'expandable/sdcard,available'),
            ('Audio/Sound/Speaker', 'front/earpiece/call,back/loud,music/song,earphone/headphone'),
            ('Build/Design','metal/plastic/glass'),
            ('Heat/temperature', '')]


# # Functions for doing the classification
# ## We use the classifier we created earlier
# ####To be run only once

# In[36]:

import language_check
tool = language_check.LanguageTool('en-US')

def score(item):
    return abs(item[1])-(len(tool.check(item[0]))/float(len(nltk.word_tokenize(item[0]))))/3.0


# In[43]:

def sentanalysis(text):
    result = {}
    complete = []

    tokens = nltk.word_tokenize(text)
    
    d = {}
    for word in tokens:
        d[word] = True
    compdist = mc.prob_classify(d)

    for label in compdist.samples():
        #print("%s: %f" % (label, compdist.prob(label)))
        complete.append(compdist.prob(label))

    sents = nltk.sent_tokenize(text)
    for sent in sents:
        sent = sent.encode('utf-8')
        tokens = nltk.word_tokenize(sent)
        #tokens = [t for t in tokens if t not in stpwrds]
        #print 'tokens: ', tokens 
        if len(set(tokens))>1:
            d = best_bigram_word_feats(tokens)
            #d = {}
            #for word in tokens:
                #d[word] = True
            dist = mc.prob_classify(d)
            #for label in dist.samples():
                #print("%s: %f" % (label, dist.prob(label)))
            result[sent] = []
            result[sent].append(dist.prob('pos'))
            result[sent].append(dist.prob('neg'))

    return (complete, result)

#E.g. of format of f: ('Camera', 'front/selfie/secondary,rear/back/primary')
def findByF(f, items):
    #print items
    subfs = f[1].split(',')
    primf = f[0]
    flist = primf.split('/')
    
    subf_items = {k:[] for k in subfs}
    general_items = []
    count = 0
    for i in items:  
        #first check if item is talking about the feature
        if any([fi.lower().strip() in i[0] for fi in flist]):
            count+=1
            #print ('\t\t\t# '+i[0])
            hasSF = False
            #find which subfeature does the item belong to
            if subfs != ['']:              
                for sf in subfs:
                    sflist = sf.split('/')
                    if any([sfi.lower().strip() in i[0] for sfi in sflist]):
                        subf_items[sf].append(i)
                        hasSF = True
                    
            if not hasSF:          
                general_items.append(i)
    
    print "(" + str(count) + "):"
    
    for sf in subf_items:
        subf_items[sf].sort(key=score, reverse=True)
        if subf_items[sf]!=[]:
            print '\t\t' + sf + ' (' + str(len(subf_items[sf])) + "):"
            for item in subf_items[sf]:
                print '\t\t\t\t> ' + item[0]
    
    if general_items != []:
        general_items.sort(key=score, reverse=True)
        print '\t\tGeneral (' + str(len(general_items)) + "):"
        for item in general_items:
            print '\t\t\t\t> ' + item[0]
    
def thefunction(mtext):
    comp, res = sentanalysis(mtext)
    for i in res:
        res[i] = res[i][0]-res[i][1]
    items = sorted(res.items(), key = lambda i: -abs(i[1]))
    pros, cons = assignPC(items,0.1)
    printByFeatures(pros, cons)

def assignPC(items, sensitivity):
    pros=[]
    cons=[]
    for i in items:
        if i[1]>sensitivity:
            pros.append(i)
        elif i[1]<(-1*sensitivity):
            cons.append(i)
    return pros,cons

def printAll(items):
    for i in items:
        print ('\t\t> '+i[0])

def printByFeatures(pros, cons):
    for f in features:
        print "---------------"
        print(f[0])
        print("\tPROS"),
        findByF(f, pros)
        print("\tCONS"),
        findByF(f, cons)
    print("\n\tALLPROS (" + str(len(pros)) + "):")
    printAll(pros)
    print("\n\tALLCONS (" + str(len(cons)) + "):")
    printAll(cons)


# # Getting input from file
# ###Run this whenever data is changed

# In[44]:

mtextfile = io.open("test.txt", "r", encoding='utf-8')
mtext = mtextfile.read()
mtextfile.close()

matches = tool.check(mtext)


mtext = cleandata(mtext)


# In[45]:

#print (mtext)
print "Language errors:", len(matches)
thefunction(mtext)


# In[ ]:



