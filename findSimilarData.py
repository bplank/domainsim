#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 20 jul 2010

Extracts train data:
- by clustering articles (find k most similar)
- or random selection

@author: Barbara Plank
"""
from optparse import OptionParser
import sys
import os
from random import sample
from Corpora import Corpora
from Corpus import Corpus

##### main stuff

usage = "usage: %prog [options] TESTFILE DIRECTORY/FILE\n \
\n\
TESTFILE: \tTarget file in CoNLL format\n\
DIRECTORY/FILE:\tDirectory with CoNLL files OR single FILE"

parser = OptionParser(usage=usage)

# optional

parser.add_option("-s", "--use-sentences", action="store_true", default=False, dest="sentences_as_unit",
                                        help="if DIRECTORY is given this option specifies to use sentences as base unit rather than articles (default: False, articles are base units to select from DIRECTORY)\nIf a single FILE is given (rather than a DIRECTORY that contains several files), then this is True since every single sentence in FILE is considered as a separate unit")

parser.add_option("--train-data-out", action="store", type="string", dest="train_data_out",
                  help="path to directory where to store selected training data. It will be created if it does not exist (default: traindata)", default="traindata")

parser.add_option("-m", "--similarity-metric", action="store", type="string", dest="similarity_metric",
                                        help="(default: jensen-shannon) "
                                        "Other options: skew, cosine, ran (=random), renyi, variational, euclidean",default="jensen-shannon")

parser.add_option("--alpha", dest='alpha', 
                                        type="float",
                                        default=0.99,
                                        help='Parameter for the skew or renyi divergence (default: 0.99)')


parser.add_option("--use-character-ngrams", dest='use_cngrams',
                                        action='store_true',
                                        default=False,
                                        help='use rel.freq. of character 4-grams (default: relative freq. of words)')

parser.add_option("--use-topicmodel", dest='use_topicmodel',
                                        action='store_true',
                                        default=False,
                                        help='use topic distribution estimated by topicmodel (default: relative freq. of words)')

parser.add_option("--mallet-topicmodel", dest='path_topicmodel',
                                        type="string",
                                        help='file with mallet document-topic-distribution')

parser.add_option("--remove-stopwords", dest='remove_stopwords',
                                        action='store_true',
                                        default=False,
                                        help='remove stopwords (default: False)')

parser.add_option("--language", dest='language',
                                        action='store',
                                        type="string",
                                        default="EN",
                                        help='language (for stopword removal: NL or EN; default is EN)')

parser.add_option("--step-size", dest='stepSize',
                                        type="int",
                                        help='Step size (number of articles to add in each round; if --step-unit-sents then this is the threshold in terms of number of sentences to select in each round)')

parser.add_option("--max", dest='max',
                                        type="int",
                                        default=10,
                                        help='Maximum number of articles (or sentences if --step-unit-sents is specified) to select (default: 10)')


parser.add_option("--step-unit-sents", dest='selectByNumSents',
                                        action='store_true',
                                        default=False,
                                        help='select by num sents rather than number k of articles (default: False) - this is the case if --use-sentences activated or single FILE given')


parser.add_option("--use-top-k", dest='top_k',
                                        type="int",
                                        help='Use only top k features (top k words/n-grams, i.e. k most frequent across corpora)')



parser.add_option("--format", dest="format",
                                        action='store',
                                        type="string",
                                        default="CoNLL",
                                        help="format of files (default: CoNLL). Alternative: plain (still experimental)")



def main():

    (options, args) = parser.parse_args()
    env = os.environ.get("MEASURES_HOME")
    if env == "":
        print("Please set MEASURES_HOME!")
        sys.exit(-1)
    
    if len(args) != 2:
        sys.stderr.write('Error: incorrect number of arguments\nSee -h\n')
        parser.print_help()
        sys.exit(-1)
    else:

        testFileName = args[0]
        testCorpus = Corpus(testFileName,format=options.format,options=options)
     
        directory = args[1]
     
        parameter=None

        # create output dir if it does not exist
        if not os.path.exists(options.train_data_out):
            os.makedirs(options.train_data_out)


        print("# similarity_metric: {0}".format(options.similarity_metric))

        sim = options.similarity_metric

        #### sim functions that have a parameter
        if options.similarity_metric == "skew" or options.similarity_metric == "renyi":
            print("# parameter: {0:f}".format(options.alpha))
            sim += str(options.alpha)
            parameter = options.alpha

        features = "w"
        useCharNgrams = options.use_cngrams # means that we use relative freq of words; 
        # if true, we use relf freqs of character 4-grams (tetragrams)    
        #print("# Using character ngrams ? {0} (if True: then we use character 4-grams)".format(useCharNgrams))
        if useCharNgrams:
            print("# Using character 4grams")

        useTopicmodel = options.use_topicmodel
        pathTopicmodel = options.path_topicmodel
        #print("# Using topicmodel ? {0} (if False: default is to use rel.freq of words)".format(useTopicmodel))
        if useTopicmodel:
            print("# Using topicmodel file: {0}".format(pathTopicmodel))

        useTopK=False # false by default
        numK=None
        if options.top_k:
            print("# Using top-k:",options.top_k)
            numK=options.top_k 
        
        print("# Loading corpora...")
        c = Corpora(directory,format=options.format,options=options,targetCorpus=testCorpus)

        keys = list(c._corpora.keys())
        key=keys[-1]
        #print("# c:", c._corpora[key].vocab)


        if useCharNgrams == True:
            features = "cn"

        if useTopicmodel == True:
            features = "tm"
            sim = "topicmodel."+options.similarity_metric

        if options.top_k:
            features += ".top"+str(options.top_k)

        if options.remove_stopwords:
            print("# Language :", options.language, " - stopwords were removed")
            features += ".nosw" # no stopwords


        print("# Corpora loaded. ")
        total = c.getNumFiles()
        print("# Total articles/sentences: ",total)

        if not options.selectByNumSents:
            maxSelect=c.getNumFiles()
        else:
            maxSelect=c.getTotalNumSents()
        if options.max < maxSelect:
            maxSelect=options.max
        
        if options.stepSize and options.max < options.stepSize:
            stepsize = options.max
            print("# Max is less than stepsize! Take max")
        elif options.stepSize:
            stepsize = options.stepSize
            print("# stepsize: {}".format(stepsize))
        else:
            stepsize=maxSelect

        steps = myrange(stepsize,maxSelect)
        if not maxSelect in steps:
            steps.append(maxSelect)


        print("# Max = {}".format(maxSelect))

        if options.similarity_metric == "ran": 
            # random selection
            
            print("# Random selection")

            randomFiles= [] # increase pool of randomly selected files
            numSentsRanSel = 0 # keep number of sents of randomly selected files

            for k in steps:
                print("# === select_k=", k)
                select_k=k
                
                if options.selectByNumSents:
                    numSentsRanSel = c.getRandomFilesWithoutReplacementUpToMaxNumSents(select_k,testCorpus.getFileName(),randomFiles,numSentsRanSel)
                    filename = testCorpus.getFileName() + ".ran.s"+str(k)
                    print(filename+"=" +' '.join(randomFiles))
                    c.saveTrainingData(filename,randomFiles,options.train_data_out)
                else:
                    c.getRandomFilesWithoutReplacement(select_k,testCorpus.getFileName(),randomFiles)
                    filename = testCorpus.getFileName() + ".ran.k"+str(k)
                    print(filename+"=" +' '.join(randomFiles))
                    c.saveTrainingData(filename,randomFiles,options.train_data_out)
        else:
            prefix=".k"
            if  options.selectByNumSents:
                print("# Selection by num sents")
                prefix=".s"
            print("# Max-s = {}".format(maxSelect))
            print("# Saving train data in directory: {}".format(options.train_data_out))

            print("# testFile:",testCorpus.getFileName())
                      
            for k in steps:
                print("# === select_k=", k)
                select_k=k
                
                ranking = c.getKMostSimilarArticles(testCorpus, select_k,options=options,parameter=parameter)
                filen = testCorpus.getFileName() + "."+features+ "."+sim+prefix+str(select_k)
                print(filen+"=" +' '.join(ranking)) #k-most-similar
                c.saveTrainingData(filen,ranking,options.train_data_out)

            
                


def myrange(start, end):
    """ same as the python range function, but stepsize is doubled (start, start *2, start * 2 * 2, etc...). Note: it excludes END
    """
    current = start
    list = [current]
    while ((current*2) < end):
        current = current * 2
        list.append(current)
    return list


main()
