import subprocess
import tempfile
import shutil
import sys
import os

class Document:
    """ represents a document as topic-distribution"""
    def __init__(self, source, topicDistribution):
        self.source = source
        self.filename = self.findFileName(source)
        self.topicDistribution = topicDistribution
        self.numTopics = len(topicDistribution.keys())

    def findFileName(self, source):
        splitted = source.split("/")
        filename = splitted[-1]
        return filename

    def printDistr(self):
        print("len: {}".format(len(self.topicDistribution)))
        for x in self.topicDistribution:
            print("{0} {1}".format(x, self.topicDistribution[x]))

    def getTopicDistribution(self):
        return self.topicDistribution


class Topicmodel:
    """ a topic model estimated with mallet """
    def __init__(self):
        self.documents = []
        self.tmpDir="/dev/shm"

    def setFromFile(self,filename):
        self.filename = filename
        self.topicmodel = self._readMalletFile(filename)

    def addDocument(self,document):
        self.documents.append(document)

    def getNumTopics(self):
        if len(self.documents)>0:
            doc = self.documents[0]
            return doc.numTopics
        else:
            return 0
    
    def getDocument(self,filename):
        for d in self.documents:
            if d.filename == filename:
                return d
        print("Document {0} not found.".format(filename),file=sys.stderr)
        sys.exit()

    def _readMalletFile(self,fileName):
        """ read mallet file and construct TopicModel object
        #doc source topic proportion ...
        (first line contains this comment)
        """
        topicModel = [] # list of topic distributions
        FILE = open(fileName)
        for l in FILE:
            l = l.strip()
            if not l.startswith("#"):
                lineArray = l.split(" ")
                docno = lineArray[0]
                source = lineArray[1]
                i=0
                topicDistr = {}
                topic = "init"
                for item in lineArray[2:]:
                    if i % 2 == 0:
                        topic = item
                    else:
                        proportion = float(item)
                        if proportion != 0.0: #only add if value diff from 0.0 === no add all
                            topicDistr[topic] = proportion
                    i+=1
                document = Document(source, topicDistr)
                self.addDocument(document)
        FILE.close()
        return topicModel


    def estimate(self,targetCorpus,directory,fileNames):
        """ estimate Mallet topic model - article level """
        print('# (experimental code: use --mallet-topicmodel) ')
        d = tempfile.mkdtemp(prefix='tmp',dir=self.tmpDir)
        # create temporary directory with files
        shutil.copy(targetCorpus.getPath(),d)
        for fileName in fileNames:
            path = directory + "/" + fileName
            shutil.copy(path,d)
        d_sents = tempfile.mkdtemp(prefix='tmp',dir=self.tmpDir)
        print("# tmp file: ",d_sents)
        for f in os.listdir(d):
            filein = d + "/" + f
            fileout = d_sents + "/" + f
            subprocess.call("cat "+ filein + " | awk '{if (NF>0) printf \"%s \",$2; else print}' > " + fileout,shell=True)
        shutil.rmtree(d)
        cdir = subprocess.call("pwd",shell=True)
        self.__callMallet(directory,d_sents)

    def __callMallet(self,directory,d_sents):
        
        myhome = os.environ.get("MEASURES_HOME")
        if directory.endswith("/"):
            directory = directory[:-1]
        subprocess.call(myhome+"/mallet-2.0.6/bin/mallet import-dir --input "+d_sents+" --output "+directory+".mallet --keep-sequence",shell=True)
        subprocess.call(myhome+"/mallet-2.0.6/bin/mallet train-topics --input "+directory+".mallet --num-topics 100 --output-state "+directory+".mallet.state.gz --output-doc-topics "+directory+".mallet.doc-topics.gz --output-topic-keys "+directory+".mallet.topic-keys.gz",shell=True)
        print("# mallet file saved in: ",directory+".mallet.doc-topics.gz")
        shutil.rmtree(d_sents)
        self.setFromFile(directory+".mallet.doc-topics.gz")

    def estimateFromSents(self,targetCorpus,corpora):
        """ estimate Mallet topic model - every sentence is a doc """
        d = tempfile.mkdtemp(prefix='tmp',dir=self.tmpDir)
        print("# Temp dir: ",d)
        for corpusName in corpora:
            corpus = corpora[corpusName]
            FILE = open(d+"/"+corpusName,"w")
            for instance in corpus.getInstances():
                sent = " ".join(instance.getSentence()) + "\n"
                FILE.write(sent)
            FILE.close()
        shutil.copy(targetCorpus.getPath(),d)
        self.__callMallet("topicmodel",d)
        


def main():
    test="/net/aistaff/bplank/data/bplank/english/topicmodels/wsj-w.doc-topics.gz"
    t = Topicmodel(test)
    print(t.getNumTopics())
#    d = t.documents[0] #get first document
#    d.printDistr()
#    print(d.filename)
#    d2 = t.documents[1]
#    d2.printDistr()
#    print(d2.filename)

    d3 = t.getDocument("wsj_2300")
    d4 = t.getDocument("wsj_2168") # js: 0.137698
    d5 = t.getDocument("wsj_1208") # js: 0.182491
    #d3.printDistr()
    print(d3.filename)
    d3.printDistr()
    print(d4.filename)
    d4.printDistr()
    print(d5.filename)
    d5.printDistr()
    subprocess.call("mallet")
#main()
