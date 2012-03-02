#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from numpy import zeros, dot
from numpy.linalg import norm
from math import log, fabs, sqrt
from conll.Conll07Reader import Conll07Reader

class MyError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class Corpus:
    """
    Represents a corpus 
    """
    _sent_delimiter = " "
    _path_delimiter = "/"
    _remove_stopwords = False # by default false

    _num_tokens = 0
    _num_sents = 0 
    _base = 2 #if base=0, natural logarithm (base e) will be taken
    """ similarity functions """
    _sim_jensen_shannon = "jensen-shannon"
    _sim_renyi = "renyi"
    _sim_skew = "skew"
    _sim_cosine = "cosine"
    _sim_variational = "variational"
    _sim_euclidean = "euclidean"
    
    options = None

    def __init__(self,path,format="CoNLL",options=None,tm=None,instance=None):
        self.path = path #path+filename
        pathList = self.path.split(Corpus._path_delimiter)
        self.filename = pathList[-1]

        self._relfreq = {}
        self.sentences = []
        self.instances = [] #Dependency Instances (if CoNLL format)
        
        if options:
            self.options = options

        if options and options.remove_stopwords:
            self._stopwords = self.loadStopwords("stopwords/"+options.language+".txt")
        else:
            self._stopwords = [] #empty
     
        if not instance:
            self.sentences = self.loadFile(path,format)
        else:
            newSentence = []
            for word in instance.getSentence():
                word = word.lower()
                if self.options and self.options.remove_stopwords:
                    if not self.isStopword(word):
                        newSentence.append(word)
                else:
                    newSentence.append(word)
            self.sentences.append(newSentence)
            self.instances = [instance]

        if options and options.use_cngrams:
            self.vocab = self.extractCharacterNGrams()
        elif options and options.use_topicmodel and tm:
            doc = tm.getDocument(self.filename)
            self.vocab = doc.getTopicDistribution()  ## not that for TM vocab contains proportions, not counts!
            self._relfreq = doc.getTopicDistribution()
            if options and options.selectByNumSents:
                numTokens, numSents = self.extractNumTokensSentences()
                self._num_tokens = numTokens 
                self._num_sents = numSents
        else:
            self.vocab = self.extractVocab()
        self.labels = [] # initialize to empty list

    def loadStopwords(self,filename):
        """ load stopwords from file """
        try:
            stopwords = []
            FILE = open(filename,"r")
            for l in FILE:
                l = l.strip()
                stopwords.append(l)
            #print("# {0} stopwords loaded.".format(len(stopwords)),file=sys.stderr)
            return stopwords
        except Exception as errStr:
            print("An error occured when loading stopword list: {}".format(errStr),file=sys.stderr)
            sys.exit(-1)

    def loadFile(self,path,format):
        """ load file and store sentences in list of lists 
            - converts all words to lowercase
        """
        sentences = [] 
        if format=="plain":
            FILE = open(path,"r")
            for l in FILE:
                l = l.strip()
                if not len(l) == 0 and l != "\n": #ignore blank lines:
                    sentence = l.split(Corpus._sent_delimiter)
                    newSentence = []
                    for word in sentence:
                        word = word.lower()
                        if self.options and self.options.remove_stopwords:
                            if not self.isStopword(word):
                                newSentence.append(word)
                        else:
                            newSentence.append(word)
                    sentences.append(newSentence)
            FILE.close()
        elif format =="CoNLL":
            reader = Conll07Reader(path)
            ### keep instances!!
            self.instances = reader.getInstances()
            for instance in self.instances: #for sentence in reader.getSentences():
                sentence = instance.getSentence()
                #sentence = list(map(lambda x: x.lower(),sentence))
                newSentence = []
                for word in sentence:
                    word = word.lower()
                    if self.options and self.options.remove_stopwords:
                        if not self.isStopword(word):
                            newSentence.append(word)
                    else:
                        newSentence.append(word)
                sentences.append(newSentence)
        #print("sentences:",sentences)
        return sentences
    
    def isStopword(self,word):
        """ Return true if word is in stopword list """
        return word.lower() in self._stopwords
    
    def getFileName(self):
        """ return filename """
        return self.filename
    
    def getPath(self):
        """ return entire path (includes filename) """
        return self.path

    def getInstances(self):
        return self.instances
        
    def updateVocab(self, newVocab, newRelFreq):
        self.vocab = newVocab 
        self._relfreq = newRelFreq
        self._num_tokens = len(newRelFreq.keys())

 
    def extractNumTokensSentences(self):
        ## returns num tokens and num sents in file (used for topicmodels)
        numTokens = 0
        for sentence in self.sentences:
            numTokens += len(sentence)
        return numTokens, len(self.sentences)

    def extractVocab(self):
        """ Extract vocabulary:
        - optionally remove stopwords
        
        >>> corpusA = Corpus("./test/a.txt",format="plain")
        >>> corpusA.vocab
        {'a': 1, 'c': 2, 'b': 1}
        >>> corpusB = Corpus("./test/b.txt",format="plain")
        >>> corpusB.vocab
        {'a': 5, 'c': 1, 'b': 2}
        >>> c = Corpus("./test/g7964483.conll",format="CoNLL")
        >>> c.getNumSents()
        8
        """
        words = {} # store in hash table
        numTokens = 0
        numSents = 0

        for sentence in self.sentences:
            numSents+=1
            for currentWord in sentence:
                numTokens +=1
                if not currentWord in words:
                    words[currentWord] = 1
                else:
                    words[currentWord] = words[currentWord] + 1  

        self._num_tokens = numTokens 
        self._num_sents = numSents
        for w in words:
            self._relfreq[w] = int(words[w]) / float(self._num_tokens) 
        return words
    
    def extractCharacterNGrams(self,ngramsize=4):
        """ Extract character n-grams on lowercase words:
        - keep \n (don't strip away end of sentence marker!)
        """
        ngrams = {} # store in hash table
        numNgramTokens = 0
        numSents = 0
           
        for sentence in self.sentences:
            numSents+=1
            line_ngrams = self.cngram(" ".join(sentence),ngramsize)
            for currentNgram in line_ngrams:
                        numNgramTokens += 1
                        if not currentNgram in ngrams:
                            ngrams[currentNgram] = 1
                        else:
                            ngrams[currentNgram] = ngrams[currentNgram] + 1 
        self._num_tokens = numNgramTokens
        self._num_sents = numSents
        for w in ngrams:
            self._relfreq[w] = int(ngrams[w]) / float(self._num_tokens) 
        return ngrams

    
    def cngram(self,string,ngramsize=4):
        ngrams = []
        l = len(string) + (ngramsize-2)
        string = " "*(ngramsize-1) + string + " "*(ngramsize-1)
        for i in range(l):    
            ngram = string[i:i+ngramsize]
            ngrams.append(ngram)
        return ngrams
    
    def getNumTokens(self):
        """ Returns number of tokens in corpus
        
        >>> corpusA = Corpus("./test/a.txt",format="plain")
        >>> corpusA.getNumTokens()
        4
        """
        return self._num_tokens
    
    def getNumSents(self):
        """ Returns number of sentences (lines) in corpus
        
        >>> corpusA = Corpus("./test/a.txt",format="plain")
        >>> corpusA.getNumSents()
        3
        >>> corpusX = Corpus("./test/x.txt",format="plain")
        >>> corpusX.getNumSents()
        4
        """
        return self._num_sents
    
    def getFileName(self):
        return self.filename
    
    def getPath(self):
        return self.path

    def getLog(self, number):
        """ Returns log to base self._base """
        try:
            if self._base != 0:
                return log(number,self._base)
            else:
                return log(number)
        except ValueError:
            print("This is not a valid number: {}".format(number),file=sys.stderr)           
                                   
    def kl_div(self, b_corpus):
        """ Returns KL-divergence of corpus to another corpus (b_corpus). 
        * is 0 if corpora are the same
        * is non-symmetric, i.e. a.kl_div(b) != b.kl_div(a)
        Note: KL is only defined it corpora contain same items (words), since we do not smooth freqs
        
        >>> corpusA = Corpus("./test/a.txt",format="plain")
        >>> corpusB = Corpus("./test/b.txt",format="plain")
        >>> corpusA.kl_div(corpusA)
        0.0
        >>> corpusA._relfreq
        {'a': 0.25, 'c': 0.5, 'b': 0.25}
        >>> corpusB._relfreq
        {'a': 0.625, 'c': 0.125, 'b': 0.25}
        >>> corpusA.kl_div(corpusB)
        0.6695179762781595
        >>> corpusB.kl_div(corpusA)
        0.5762050593046013
        """
        #get set of words from both corpora
        keys = set(list(self.vocab.keys()) + list(b_corpus.vocab.keys()))
        sum = 0
        #  calculate sum q(y) (log q(y) - log q(r))
        for w in keys:
            q_y = 0
            r_y = 0
            
            #if w not in self._relfreq or w not in b_corpus._relfreq:
            if w not in b_corpus._relfreq:
                try:
                    raise MyError(w)
                except MyError:
                    print("My exception occurred. Word is not in b_corpus (r(y)=0).")
                    print("Cannot calculate KL-div: word is not in r! Return -1.")
                    return -1
            else:
                q_y = self._relfreq[w]
                r_y = b_corpus._relfreq[w]
                sum += (q_y * (self.getLog(q_y) - self.getLog(r_y)))   
        return sum
    
    def jensen_shannon(self, b_corpus):
        """ Returns Jensen-Shannon divergence. cf. (Lee, 2001)
        * Jensen-Shannon is symmetric
        * It considers the KL-div between q, r, and the average of q and r
        
        >>> corpusA = Corpus("./test/a.txt",format="plain")
        >>> corpusB = Corpus("./test/b.txt",format="plain")
        >>> corpusBc1 = Corpus("./test/b-c1.txt",format="plain")
        >>> corpusX = Corpus("./test/x.txt",format="plain")
        >>> corpusA.jensen_shannon(corpusB)
        0.1467822215997982
        >>> corpusB.jensen_shannon(corpusA)
        0.1467822215997982
        >>> corpusA.jensen_shannon(corpusA)
        0.0
        >>> corpusA.jensen_shannon(corpusBc1)
        0.3723847512520989
        >>> corpusBc1.jensen_shannon(corpusA)
        0.3723847512520989
        >>> corpusA.jensen_shannon(corpusX)
        0.7126417936566769
        """ 
        #get set of words from both corpora
        keys = set(list(self.vocab.keys()) + list(b_corpus.vocab.keys()))
        #keys = set(self.vocab.keys())
        sum = 0.0 # final sum
        sum1 = 0.0
        sum2 = 0.0
        for w in keys:
            q_y = 0.0
            r_y = 0.0
            if w in self.vocab:
                q_y = self._relfreq[w]
            if w in b_corpus.vocab:
                r_y = b_corpus._relfreq[w]
            
            avg = (q_y + r_y) * 0.5
            
            if w in self.vocab:
                sum1 += (q_y * (self.getLog(q_y) - self.getLog(avg)))   
            if w in b_corpus.vocab:
                sum2 += (r_y * (self.getLog(r_y) - self.getLog(avg)))
        #identical implementation: 
        # entropy of average - [( entropy p + entropy r) / 2 ]
        sum = (sum1 + sum2) * 0.5   
        return sum
    
    def skew_div(self, b_corpus,alpha=0.5):
        """ Returns skew divergence. cf. (Lee, 2001)
        * Is asymmetric (non-symmetric)
        * smooths one of the distributions by mixing it, to a degree defined by alpha, with the other distr
        * 0 iff distributions are equal
        * always defined, even if on disjoint event sets (as long as alpha < 1). Then it gets a large number.
        
        >>> corpusA = Corpus("./test/a.txt",format="plain")
        >>> corpusB = Corpus("./test/b.txt",format="plain")
        >>> corpusA.skew_div(corpusB)
        0.13719722204191787
        >>> corpusB.skew_div(corpusA)
        0.15636722115767854
        >>> corpusA.skew_div(corpusA)
        0.0
        >>> a = Corpus("./test/aa.txt",format="plain")
        >>> b = Corpus("./test/bb.txt",format="plain")
        >>> a.skew_div(b,alpha=0.99)
        2.117287003573289
        >>> a.skew_div(corpusB,alpha=0.99)
        6.6438561897747235
        
        """ 
        #get set of words from corpus A only (because if 0 in A, then 0 * log... is 0
        keys = self.vocab.keys() 
        sum = 0.0 # final sum
        for w in keys:
            p_y = 0.0
            q_y = 0.0
            if w in self.vocab:
                p_y = self._relfreq[w]
            if w in b_corpus.vocab:
                q_y = b_corpus._relfreq[w]
            
            mix = (alpha * q_y) + ((1-alpha) * p_y) 
            sum += (p_y * (self.getLog(p_y) - self.getLog(mix)))        
        return sum
    
    def variational(self,b_corpus):
        """ Returns the variational metric (sum of absolute differences)        
        #    return self.getOneRandomFile(testFile, filesAlreadySelected)
        #else:
        #    return [filename]
        * is symmetric
        
        >>> corpusA = Corpus("./test/a.txt",format="plain")
        >>> corpusB = Corpus("./test/b.txt",format="plain")
        >>> corpusA.variational(corpusB)
        0.75
        >>> corpusB.variational(corpusA)
        0.75
        >>> corpusA.variational(corpusA)
        0.0
        >>> a = Corpus("./test/aa.txt",format="plain")
        >>> b = Corpus("./test/bb.txt",format="plain")
        >>> b_nooverlap = Corpus("./test/b_nooverlap.txt",format="plain")

        >>> a.variational(b)
        1.2
        >>> a.variational(b_nooverlap) # if no overlap, sum of probs
        2.0
        
        """ 
        #get set of words from both corpora
        keys = set(list(self.vocab.keys()) + list(b_corpus.vocab.keys()))

        sum = 0.0 # final sum
        for w in keys:
            q_y = 0.0
            r_y = 0.0
            if w in self.vocab:
                q_y = self._relfreq[w]
            if w in b_corpus.vocab:
                r_y = b_corpus._relfreq[w]
            sum += fabs(q_y - r_y)
        return sum
    
    def euclidean(self,b_corpus):
        """ Returns the euclidean distance  
        squared root of sum of squared distances
        >>> corpusA = Corpus("./test/a.txt",format="plain")
        >>> corpusB = Corpus("./test/b.txt",format="plain")
        >>> corpusA.euclidean(corpusB)
        0.5303300858899106
        >>> corpusB.euclidean(corpusA)
        0.5303300858899106
        >>> corpusA.euclidean(corpusA)
        0.0
        >>> a = Corpus("./test/aa.txt",format="plain")
        >>> b = Corpus("./test/bb.txt",format="plain")
        >>> a.euclidean(b)
        0.5656854249492381
        >>> b_nooverlap = Corpus("./test/b_nooverlap.txt",format="plain")
        >>> a.euclidean(b_nooverlap)
        1.019803902718557
        """ 
        #get set of words from both corpora
        keys = set(list(self.vocab.keys()) + list(b_corpus.vocab.keys()))
        sum = 0.0 # final sum
        for w in keys:
            q_y = 0.0
            r_y = 0.0
            if w in self.vocab:
                q_y = self._relfreq[w]
            if w in b_corpus.vocab:
                r_y = b_corpus._relfreq[w]
                
            sum += (q_y - r_y) ** 2
        sum = sqrt(sum)
        return sum
    
    def renyi(self, b_corpus,alpha=0.5):
        """ Returns the Renyi divergence. cf. (van Asch and Daelemans, 2010)
        * Renyi is parametrized by alpha (kind of a mixing weight)
        * With alpha = 1 Renyi would (in the limit) tend to the KL (note: Renyi is not defined for alpha = 1)
        * is symmetric (contrary to what van Asch writes??)
        * returns infinity if corpora are disjoint
        
        >>> corpusA = Corpus("./test/a.txt",format="plain")
        >>> corpusB = Corpus("./test/b.txt",format="plain")
        >>> corpusA.renyi(corpusB,0.5)
        0.3191631025370074
        >>> corpusB.renyi(corpusA,0.5)
        0.3191631025370074
        >>> corpusA.renyi(corpusA,0.5)
        0.0
        >>> corpusD = Corpus("./test/d.txt",format="plain")
        >>> corpusA.renyi(corpusD,0.5)
        5.087462841250339
        >>> corpusA.renyi(corpusA,0.5)
        0.0
        >>> corpusNoOverlap = Corpus("./test/dutch.txt",format="plain")
        >>> corpusA.renyi(corpusNoOverlap,0.5)
        inf
        """
        ### check: if they do not overlap, return infinity (renyi is not defined)
        if (set(self.vocab.keys())).isdisjoint(set(b_corpus.vocab.keys())):
            return float('infinity')

        #get set of words from both corpora
        keys = set(list(self.vocab.keys()) + list(b_corpus.vocab.keys()))        
        sum = 0.0
        p_k = 0.0
        q_k = 0.0
        for w in keys:
            # like Vincent: compute only over words in both corpora (asymetric!)
            if w in self._relfreq:
                p_k =  self._relfreq[w]
                if  w in b_corpus._relfreq:
                    q_k = b_corpus._relfreq[w]
                    #sum += (p_k**(1-alpha) * q_k**alpha) #(vincent) wrong?
                    sum += (p_k**alpha) * (q_k**(1-alpha))  #(wikipedia)

        renyi = (1/float(alpha-1)) * self.getLog(sum)
        if renyi == -0.0:
            return 0.0
        else:
            return renyi
    
    def overlap_all(self, b_corpus):
        """ Returns the plain lexical overlap of all items in vocab
        
        >>> corpusA = Corpus("./test/a.txt",format="plain")
        >>> corpusB = Corpus("./test/b.txt",format="plain")
        >>> corpusX = Corpus("./test/x.txt",format="plain")
        >>> corpusA.overlap_all(corpusB)
        3
        >>> corpusA.overlap_all(corpusX)
        1
        >>> corpusX.overlap_all(corpusA)
        1
        """
        overlap=0 
        #get set of words from both corpora
        for w in self.vocab.keys():
            if w in b_corpus.vocab.keys():
                overlap+=1
        return overlap
    
    def toVector(self, keys):
        vector = zeros(len(keys))
        for index, word in enumerate(keys):
            if word in self.vocab:
                vector[index] = self._relfreq[word]
            else:
                vector[index] = 0.0
        return vector
    
    def _getCosineSimilarity(self,v1,v2):
        return float(dot(v1, v2) / (norm(v1) * norm(v2)))
    
    def cosine_sim(self, b_corpus):
        """ Returns cosine-similarity of corpus to another corpus (b_corpus).
        Maximum Similarity: 1 (if two corpora are the same) 
        
        >>> corpusA = Corpus("./test/a.txt",format="plain")
        >>> corpusB = Corpus("./test/b.txt",format="plain")
        >>> corpusA.cosine_sim(corpusA)
        1.0000000000000002
        >>> corpusA.cosine_sim(corpusB)
        0.6708203932499369
        >>> corpusB.cosine_sim(corpusA)
        0.6708203932499369
        >>> corpusAdoubled = Corpus("./test/a-doubled.txt",format="plain")
        >>> corpusAdoubled.cosine_sim(corpusAdoubled)
        1.0000000000000002
        >>> corpusAdoubled.cosine_sim(corpusA)
        1.0000000000000002
        >>> corpusB.cosine_sim(corpusAdoubled)
        0.6708203932499369
        
        """
        #get set of words from both corpora
        keys = set(list(self.vocab.keys()) + list(b_corpus.vocab.keys()))
        
        vectorA = self.toVector(keys) 
        vectorB = b_corpus.toVector(keys)
        return self._getCosineSimilarity(vectorA, vectorB)
            
    
###### main stuff

if __name__ == "__main__":
    import doctest
    doctest.testmod()

