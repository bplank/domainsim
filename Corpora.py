'''
Created on 14 jun 2010

@author: p252438
'''
#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import gzip

from Corpus import Corpus
from Corpus import MyError
from random import randint
from operator import itemgetter
from Topicmodel import Topicmodel
from myconllutils.conll.Conll07Reader import Conll07Reader

class Corpora:
    """ A collection of Corpus objects
    """
    
    def __init__(self,pool,format="CoNLL",options=None,tm=None,targetCorpus=None):
        """ initalize
        >>> c = Corpora("test/testdir",format="plain")
        >>> c._files
        ['a.txt', 'b.txt', 'b-c1.txt']
        >>> c.getNumFiles()
        3
        >>> c.getNumCorpora()
        3
        >>> c1 = c._corpora['a.txt']
        >>> c1.getFileName()
        'a.txt'
        >>> c1.getPath()
        'test/testdir/a.txt'
        >>> c1._relfreq
        {'a': 0.25, 'c': 0.5, 'b': 0.25}
        >>> c._files[0]
        'a.txt'
        >>> c.getKMostSimilarArticles(c1,size=1)
        ['b.txt@0.146782']
        >>> x = c._corpora['b-c1.txt']
        >>> x2 = c._corpora['b.txt']
        >>> x.jensen_shannon(x2)
        0.125
        
        """
        self.articles_as_unit=True
        if options and options.sentences_as_unit:
            self.articles_as_unit=False
        self._corpora = self.loadCorpora(pool,format=format,options=options,tm=tm,target=targetCorpus)
        if options and options.top_k:
            # restrict vocabulary of corpus to top K
            restrictedVocab = self.getTopK(options.top_k)
            print("# restrictedVocab: {}".format(restrictedVocab), file=sys.stderr)
            for corpus in self._corpora:
                newRelFreq = {}
                newVocab = {}
                # only get words from newVocab
                for word in restrictedVocab:
                    if word in corpus._relfreq:
                        newRelFreq[word] = corpus._relfreq[word]
                        newVocab[word] = corpus.vocab[word]
                corpus.updateVocab(newVocab,newRelFreq)

        sys.setrecursionlimit(self.getNumFiles())
                

    def loadFiles(self, directory,target=None):
        """ load names of files from directory into list, 
            excluding target if present """
        fileList = []
        for f in os.listdir(directory):
            if target:
                if not f == target.getFileName():
                    fileList.append(f)
            else:
                fileList.append(f)
        return fileList

    def getNumFiles(self):
        return len(self._corpora.keys())

    def getFileNames(self):
        return list(self._corpora.keys())

    def loadCorpora(self,pool,format="CoNLL",options=None,tm=None,target=None):
        """ load Corpora from files """
        self.totalNumSents = 0
        corpora = {}
        self._files = []

        # if it is a directory
        if os.path.isdir(pool) and self.articles_as_unit:
            self._files = self.loadFiles(pool,target=target) 
            self._directory = pool
          
            if options and options.use_topicmodel:
                print("# loading topicmodel from: ",options.path_topicmodel)
                topicmodel = Topicmodel()
                if options.path_topicmodel:
                    topicmodel.setFromFile(options.path_topicmodel)
                else:
                    print("# --> estimate:")
                    topicmodel.estimate(target,self._directory,self._files)
                for d in topicmodel.documents:
                    f = d.filename
                    if target and f == target.getFileName():
                        continue
                    self._files.append(d.filename)
                    corpus = Corpus(pool+Corpus._path_delimiter+d.filename,format=format,options=options,tm=topicmodel)
                    corpus.updateVocab(d.getTopicDistribution(),d.getTopicDistribution())
                    corpora[corpus.getFileName()] = corpus
            else:
                for f in self._files:
                    corpus = Corpus(pool+Corpus._path_delimiter+f,format=format,options=options)
                    self.totalNumSents += corpus.getNumSents()
                    corpora[corpus.getFileName()] = corpus

        elif os.path.isdir(pool) and not self.articles_as_unit:
            print("# --> use sentences as unit")
            if options and options.use_topicmodel:
                print("Sorry, topicmodel not yet implemented for sentences!")
                sys.exit(-1)
            tmpFiles = self.loadFiles(pool,target=target)
            self._directory = pool
            for myFile in tmpFiles:
                reader = Conll07Reader(pool+Corpus._path_delimiter+myFile)
                count=1 #count lines of file, start with 1 (like cat -n)
                instances = reader.getInstances()
                #for sentence in reader.getSentences():
                for instance in instances:
                    sentence = instance.getSentence()
                    filename=myFile+":s:"+str(count)
                    corpus = Corpus(filename,format="CoNLL",options=options,instance=instance)
                    self.totalNumSents += corpus.getNumSents()
                    self._files.append(filename)
                    corpora[corpus.getFileName()] = corpus
                    count+=1
        else:
            self.articles_as_unit=False
            if target and pool == target.getPath():
                print("Target and FILE are the same!")
                sys.exit(-1)
            print("# single file!")
            if options and options.use_topicmodel:
                print("Sorry, topicmodel not yet implemented for sentences!")
                sys.exit(-1)
            reader = Conll07Reader(pool)
            prefix,separator,nameFile = pool.rpartition(Corpus._path_delimiter)
            self._directory=prefix
            count=1 #start count at 1
            for instance in reader.getInstances():
                sentence = instance.getSentence()
                filename=nameFile+":s:"+str(count)
                corpus = Corpus(filename,format="CoNLL",options=options,instance=instance)
                self._files.append(corpus.getFileName())
                self.totalNumSents += corpus.getNumSents()
                corpora[corpus.getFileName()] = corpus
                count+=1

        return corpora

    
    
    def getNumCorpora(self):
        return len(self._corpora)

    def getTotalNumSents(self):
        return self.totalNumSents
    
    def getCorpusSizeByFilename(self, corpusFileName):
        corpus = self._corpora[corpusFileName]
        return corpus.getNumSents()
                        
    def getTopK(self, numK):
        """ Returns the top k vocabulary items (k most frequent) across corpora
        in case of ties (words with same count), it takes them in alphabetical order
        (impl. detail: vocab is a hash with word as index and count as value)
        >>> c = Corpora("test/testdir",format="plain")
        >>> c.getNumFiles()
        3
        >>> c.getTopK(2)
        ['a', 'b']
        
        was before: {'a': 1.5, 'b': 0.75}
        """
        entireVocab = {}
        for key_corpus in self._corpora:
            corpus = self._corpora[key_corpus]
            for word in corpus._relfreq:
                if not word in entireVocab:
                    entireVocab[word] = corpus._relfreq[word]
                else:
                    entireVocab[word] += corpus._relfreq[word]
        
        listOfTuples = []
        for word in entireVocab:
            t = word, entireVocab[word] #store word, count
            listOfTuples.append(t)
        rankedByCount = sorted(listOfTuples,key=itemgetter(1),reverse=True) #sort on summed relfreq
        # convert to hash
        out = {}
        for x, y in rankedByCount[0:numK]:
            out[x] = y
        return list(out.keys())
    
    def getSimilarity(self,corpus1,corpus2,metric,parameter=None):
        """ returns the similarity score between two corpus items """
        if metric == Corpus._sim_jensen_shannon:
            val = corpus1.jensen_shannon(corpus2)
            return val
        if metric == Corpus._sim_renyi:
            if (parameter == None):
                try:
                    raise MyError
                except:
                    print("Parameter alpha not defined")
            else:
                val = corpus1.renyi(corpus2,parameter)
                return val
        if metric == Corpus._sim_skew:
            if (parameter == None):
                try:
                    raise MyError
                except:
                    print("Parameter alpha for skew div not defined!")
            else:
                val = corpus1.skew_div(corpus2,alpha=parameter)
                return val
        if metric == Corpus._sim_cosine:
            val = corpus1.cosine_sim(corpus2)
            return val
        if metric == Corpus._sim_variational:
            val = corpus1.variational(corpus2)
            return val
        if metric == Corpus._sim_euclidean:
            val = corpus1.euclidean(corpus2)
            return val
        print("Error: similarity metric is unknown: {}".format(metric),file=sys.stderr)
        sys.exit(-1)
        
        
    def getKMostSimilarArticles(self,targetCorpus,size,parameter=None,options=None):
        """
        Returns list of most similar articles plus their similarity score 
        as [filename@score, ...,filenameN@scoreN]
        """
        if options and options.similarity_metric:
            similarity=options.similarity_metric
        else:
            similarity=Corpus._sim_jensen_shannon

    
        # store elements as tuples in list: [(filename, score), ...,(filename,score)]
        listOfTuples = []
        for key_corpus in self._corpora:
            corpus = self._corpora[key_corpus]
            if not corpus.getFileName() == targetCorpus.getFileName():
                similarity_val = self.getSimilarity(targetCorpus,corpus,similarity,parameter)
                t = corpus.getFileName(), similarity_val #store filename,score
                listOfTuples.append(t)

        # return sorted list in decreasing similarity
        higherFirst = True # higher numbers reflect higher similarity
        if similarity == Corpus._sim_jensen_shannon or similarity == Corpus._sim_renyi \
           or similarity == Corpus._sim_skew or similarity == Corpus._sim_euclidean \
           or similarity == Corpus._sim_variational:
            higherFirst = False
        if similarity == Corpus._sim_cosine:
            higherFirst = True
        similarityRanking = sorted(listOfTuples,key=itemgetter(1),reverse=higherFirst) #key to sort is simScore
        topKFiles = []
        i = 0
        numSentsAlreadySelected = 0
        for simScore in similarityRanking: #x contains key =similarity value
            if options and options.selectByNumSents:
                if numSentsAlreadySelected < size:
                    item = similarityRanking[i]
                    filename = item[0]
                    numSentsAlreadySelected += self.getCorpusSizeByFilename(filename)
                    simScore = item[1]
                    filenameAndScore = "%s@%f" % (filename,simScore)
                    topKFiles.append(filenameAndScore)
                    i+=1
                else:
                    print("# totalNumSentsSelected={}".format(numSentsAlreadySelected))
                    break
            else:
                # select on article basis
                if i < size:
                    item = similarityRanking[i]
                    filename = item[0]
                    simScore = item[1]
                    filenameAndScore = "%s@%f" % (filename,simScore)
                    topKFiles.append(filenameAndScore)
                    i+=1
                else:
                    break
        return topKFiles
    
   #  def getKMostSimilarArticlesUpToMaxNumSents(self,targetCorpus, size,similarity=Corpus._sim_jensen_shannon,parameter=None):
#         """
#         Returns list of most similar articles plus their similarity score as [filename@score, ...,filenameN@scoreN]
#         """
        
#         # store elements as tuples in list: [(filename, score), ...,(filename,score)]
#         listOfTuples = []
#         for key_corpus in self._corpora:
#             corpus = self._corpora[key_corpus]
#             similarity_val = self.getSimilarity(targetCorpus,corpus,similarity,parameter)
#             t = corpus.getFileName(), similarity_val #store filename,score
#             listOfTuples.append(t)
#         # return sorted list in decreasing similarity
        
#         #item = self.simmatrix.getRowByIndex(0)
#         #itemIndex=0
#         # store elements as tuples in list: [(filename, score), ...,(filename,score)]
# #        li  stOfTuples = []
# #        for i in range(len(item)):
# #            if i != itemIndex: #exclude file itself!
# #                fileName = self._files[i]
# #                t = fileName, item[i] #store filename,score
# #                #similarityRanking[item[i]] = i
# #                listOfTuples.append(t)
# #        # return sorted list in decreasing similarity
#         higherFirst = True # higher numbers reflect higher similarity
#         if similarity == Corpus._sim_jensen_shannon or similarity == Corpus._sim_renyi \
#            or similarity == Corpus._sim_skew or similarity == Corpus._sim_perplexity or \
#            similarity == Corpus._sim_euclidean or similarity == Corpus._sim_variational:
#             higherFirst = False
#         if similarity == Corpus._sim_cosine:     
#             higherFirst = True
#         similarityRanking = sorted(listOfTuples,key=itemgetter(1),reverse=higherFirst) #key to sort is simScore
#         topKFiles = []
#         i = 0
#         numSentsAlreadySelected = 0
#         for simScore in similarityRanking: #x contains key =similarity value
#             if numSentsAlreadySelected < size:
#                 item = similarityRanking[i]
#                 filename = item[0]
#                 numSentsAlreadySelected += self.getCorpusSizeByFilename(filename)
#                 simScore = item[1]
#                 filenameAndScore = "%s@%f" % (filename,simScore)
#                 topKFiles.append(filenameAndScore)
#                 i+=1
#             else:
#                 print("# totalNumSentsSelected={}".format(numSentsAlreadySelected))
#                 break
#         return topKFiles

   
    def getOneRandomFile(self,testFile,filesAlreadySelected):
        """
        >>> c = Corpora("test/testdir",format="plain")
        >>> c.getOneRandomFile("a.txt",['b.txt'])
        ['b-c1.txt']
        """
        #size=1
        n = self.getNumFiles()
     
        index = randint(0,(n-1))
        filename = self._files[index]
        alreadySelected = (filename in filesAlreadySelected or filename == testFile)
        while alreadySelected:
            index = randint(0,(n-1))
            filename = self._files[index]
            if not filename in filesAlreadySelected and not filename == testFile:
                alreadySelected = False
                break
        return [filename]

    def getRandomFilesWithoutReplacement(self,size,testFileName,randomFilesAlreadySelected):
        while len(randomFilesAlreadySelected) < size:
            selected = self.getOneRandomFile(testFileName,randomFilesAlreadySelected)
            randomFilesAlreadySelected.extend(selected)
            if len(self._files) < size:
                try:
                    raise MyError
                except:
                    print("Error: random file selection size {0} larger than actual list of files {1}".format(size,len(self._files)))
                return []
            self.getRandomFilesWithoutReplacement(size, testFileName, randomFilesAlreadySelected)
            return randomFilesAlreadySelected
    
    def getRandomFilesWithoutReplacementUpToMaxNumSents(self,size,testFileName,randomFilesAlreadySelected,totalNumSentsSelected):
        if totalNumSentsSelected > size:
            print("# totalNumSentsSelected={}".format(totalNumSentsSelected))
            return totalNumSentsSelected
        else:
            while totalNumSentsSelected < size:
                selected = self.getOneRandomFile(testFileName,randomFilesAlreadySelected)
                filenameSelected = selected[0]
                totalNumSentsSelected += self.getCorpusSizeByFilename(filenameSelected)
                randomFilesAlreadySelected.extend(selected)
            print("# totalNumSentsSelected={}".format(totalNumSentsSelected))
        return totalNumSentsSelected

    def saveTrainingData(self,filename, listOfFilenames, destDir):
        outputname=destDir + Corpus._path_delimiter + filename + ".gz"
        FILEOUT = gzip.open(outputname,'wb')
        if self.articles_as_unit:
            for f in listOfFilenames:
                filenameList = f.split("@")
                filename = filenameList[0]
                corpus = self._corpora[filename]
                for instance in corpus.getInstances():
                    parse = instance.__repr__() + "\n"
                    FILEOUT.write(parse.encode())
        else:
            for f in listOfFilenames:
                filenameList = f.split("@")
                filenameWithLineNum = filenameList[0]
                #filename,lineNumStr = filenameWithLineNum.split(":s:")
                #lineNum = int(lineNumStr)

                #if self._directory:
                #    reader = Conll07Reader(self._directory + Corpus._path_delimiter+filename)
                #else:
                #    reader = Conll07Reader(filename)
                corpus = self._corpora[filenameWithLineNum]
                instances = corpus.getInstances()
                for instance in instances:
                    parse = instance.__repr__() + "\n"
                    FILEOUT.write(parse.encode())
                    
        FILEOUT.close()
        print("# File written: {}".format(outputname))



###### main stuff

if __name__ == "__main__":
    import doctest
    doctest.testmod()

