
# coding: utf-8

# # Preparing training Data
# ####To be run only once

# In[1]:

#prepare training data
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
    mtext = re.sub('\n+', ' ', mtext)
    #mtext = re.sub('[0-9]+. ', '', mtext)
    mtext = re.sub(' -', '.', mtext)
    mtext = re.sub('\.+', '. ', mtext)
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
 
print 'evaluating single word features'
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
 
print 'evaluating best word features'
#mc=evaluate_classifier(best_word_feats)
 
def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d
 
print 'evaluating best words + bigram chi_sq word features'
mc = evaluate_classifier(best_bigram_word_feats)


# # Functions for doing the classification
# ## We use the classifier we created earlier
# ####To be run only once

# In[3]:

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
        result[sent] = []
        #print (sent)
        tokens = nltk.word_tokenize(sent)
        #tokens = [t for t in tokens if t not in stpwrds]
        d = best_bigram_word_feats(tokens)
        #d = {}
        #for word in tokens:
            #d[word] = True
        dist = mc.prob_classify(d)
        #for label in dist.samples():
            #print("%s: %f" % (label, dist.prob(label)))
        result[sent].append(dist.prob('pos'))
        result[sent].append(dist.prob('neg'))

    return (complete, result)

def findByF(f, items):
    for i in items:
        flist = f.split('/')
        for fi in flist:
            if fi.lower().strip() in i[0]:
                print ('\t\t> '.encode('utf-8')+i[0].encode('utf-8'))
    
def thefunction(mtext):
    comp, res = sentanalysis(mtext)
    for i in res:
        res[i] = res[i][0]-res[i][1]
    items = sorted(res.items(), key = lambda i: -abs(i[1]))
    features = ['Cpu/Processor', 'Screen/Display', 'Battery', 'Camera', 'RAM/Memory']
    pros, cons = assignPC(items,0.1)
    printByFeatures(features, pros, cons)

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
        print ('\t\t> '.encode('utf-8')+i[0].encode('utf-8'))

def printByFeatures(features, pros, cons):
    for f in features:
        print(f)
        print("\tPROS:")
        findByF(f, pros)
        print("\tCONS:")
        findByF(f, cons)
    print("\n\tALLPROS:")
    printAll(pros)
    print("\tALLCONS:")
    printAll(cons)


# # Getting input from file
# ###Run this whenever data is changed

# In[4]:

mtextfile = io.open("reviewIP.txt", "r", encoding='utf-8')
mtext = mtextfile.read()
mtextfile.close()
import language_check
tool = language_check.LanguageTool('en-US')

matches = tool.check(mtext)


mtext = cleandata(mtext)


# In[5]:

#print (mtext)
print "Language errors:", len(matches)
thefunction(mtext)


# In[ ]:



