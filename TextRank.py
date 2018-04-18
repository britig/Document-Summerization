# program for implementation of TextRank for summarization of a given text document

import os
#importing nltk package for preprocessing steps
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from operator import itemgetter
import re
import math
import networkx as nx
import itertools
import pandas


tokenized_words={}             #Dictionary used for storing the tokens of sentences with associated node number(key


'''Function to tokenize the data''' 
def tokenize(data):
    sent_tokens = sent_tokenize(data)
    # list of words
    word_tokens = [word_tokenize(sent) for sent in sent_tokens]
    for k in sent_tokens:
        tokenized_words[k]=[]
        tokenized_words[k]=word_tokenize(k)
    return sent_tokens, word_tokens

'''Function to Calculate the idf score for each word
    Input : word list, number of document
    Return: word to idf dictionary
'''
def idfScoreCalculation(wordlist,N):
    wordToidf={}
    for word in wordlist:
        df = 0
        for filename in os.listdir('./Assignement2_IR/Topic2'):
            filepath = './Assignement2_IR/Topic2/' + filename
            file = open(filepath, 'r')
            text = file.read()
            if text.count(word) > 0:
                df = df + 1
                file.close()
        if df==0:
            print(filename)
            print( word)
        if word not in wordToidf.keys():
            wordToidf[word]=math.log10(N/df)
    return wordToidf

'''Function to calculate tf_idf similarity score
    Input:
    sentence1, sentence2, idf dictionary
     Returns
    -------
   cosine similarity value
'''
def tf_idf_calculation1(s1, s2, idf):
    try:
        num = 0
        comb = s1 + s2
        # Calculate the numerator for cosine similarity
        for word in comb:
            tf1 = s1.count(word)
            tf2 = s2.count(word)
            num += (int(tf1) * int(tf2) * (float(idf[word] ** 2)))
            total1 = 0
            total2 = 0
        #norm for sentence one
        for word in s1:
            tf = s1.count(word)
            total1 += ((int(tf) * float(idf[word]))**2)
        #norm for sentence 2
        for word in s2:
            tf = s2.count(word)
            total2 += ((int(tf) * float(idf[word]))**2)
        #Calculate the denominator for cosine similarity
        deno = (math.sqrt((total1))) * (math.sqrt((total2)))
        if deno == 0:
            deno = 1
        return float(num) / deno
    except Exception as e:
        print(e)

'''Function to build the graph based on certain threashold
    Input:
    sentences, threshold value, idf dictionary
     Returns
    -------
    g : Graph based on similarity value
'''
def build_graph(nodes, threshold, idf):
    g = nx.Graph()  # initialize an undirected graph
    g.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    # add edges to the graph (weighted by cosine similarity)
    for pair in nodePairs:
        node1 = pair[0]
        node2 = pair[1]
        s1=tokenized_words[node1]
        s2=tokenized_words[node2]
        simval = tf_idf_calculation1(s1, s2, idf)
        if simval > threshold:
            g.add_edge(node1, node2, weight=simval)

    return g


''' Function to calculate key sentences
    Input:
    Graph
    
    Returns
    -------
    keysentences : list
       list of nodes with sorted in descending order of pagerank'''
def get_keysentences(graph):
    # weight is the similarity value obtained from the idf_modified_cosine
    calculated_page_rank = clculate_pagerank(graph, weight='weight')
    # most important words in descending order of pagerank
    keysentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
    return keysentences

'''Pagerank specific functionalities'''

#Power iteration method for calculating pagerank
def clculate_pagerank(G, alpha=0.85, max_iter=100, tol=1.0e-6,weight='weight'):
    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G
    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()
    x = dict.fromkeys(W, 1.0 / N)
    p = dict.fromkeys(W, 1.0 / N)
    dangling_weights = p
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]
    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]
 
        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N*tol:
            return x


'''Function to read file, do preprocessing and print/write summary'''
def readFile():
    sentenceList = []
    wordTokenList = []
    wordsList = []
    docIDListSent = {}
    list_doc = os.listdir("./Assignement2_IR/Topic2")
    # stop_words = set(stopwords.words('english'))
    i = 0
    for doc in list_doc:
        file_doc = open("./Assignement2_IR/Topic2/" + str(doc), "r", encoding="utf8")
        data = file_doc.read()
        data = data.split('<TEXT>')
        data = data[1].split('</TEXT>')
        # Remove XML Tags
        data = re.sub('<[^<]+>', "", data[0])
        data=data.strip()
        # Tokenize the data into sentences and words
        sent_tokens, word_tokens = tokenize(data)
        for sent in sent_tokens:
            sentenceList.append(sent)
        for word in word_tokens:
            wordTokenList.append(word)
        for word in word_tokenize(data):
            wordsList.append(word)
        docIDListSent[i] = wordTokenList
        i = i + 1

    # Calculate the idf score
    N = 25
    idfScore = idfScoreCalculation(wordsList,N)
    print(str("here"))
    g = build_graph(sentenceList, 0.3, idfScore)
    keysentences = get_keysentences(g)
    print ("Printing Top 12 Key sentences:--------------------------\n")
    for sent in keysentences[:12]:
        print (str(sent) + "\n")

    file = open("./Assignement2_IR/Summaries/Topic20.3TR.txt", "w")
	#Break if more than 250 words
    count = 0
    for sent in keysentences[:12]:
        file.write(str(sent) + "\n")
		#limit to 250 words
        wordToken = word_tokenize(sent)
        count = count+ len(wordToken)
        if count > 250:
            break;
    file.close()


if __name__ == "__main__":
    readFile()