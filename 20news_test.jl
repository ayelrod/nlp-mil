using StatsBase
using DrWatson
using MLJ
using MLJBase
using DataFrames
using Random
using CSV
using CategoricalArrays
using PyCall

include(projectdir("minds-model-zoo/julia-models/src/wrappers/PyMISVMClassifier.jl")) 
include(projectdir("minds-model-zoo/julia-models/src/RepeatedCV.jl"))

#include("grid_search.jl")
include("pdMISVMClassifier.jl")
include("model_builders.jl")

Random.seed!(1234)

py"""

from os import listdir
from os.path import isfile, join
import string
import numpy as np
from random import seed
from random import random
from random import randint
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import gensim
import os



def parse_20newsgroups_bow(): 
    # set parameters
    num_bags = 100
    num_positive_bags = 50
    num_instances = 50 
    num_features = 200 
    positivity_rate = 0.03
    
    group = input("Choose a group: enter the name of the group - ex: alt.atheism\n")

    my_path = '20_newsgroups'
    
    #creating a list of folder names to make valid pathnames later
    folders = [f for f in listdir(my_path)]
    
    
    #creating a 2D list to store list of all files in different folders
    files = []
    for folder_name in folders:
        folder_path = join(my_path, folder_name)
        files.append([f for f in listdir(folder_path)])
    
    
    #creating a list of pathnames of all the documents
    #this would serve to split our dataset into train & test later without any bias
    pathname_list = []
    for fo in range(len(folders)):
        for fi in files[fo]:
            pathname_list.append(join(my_path, join(folders[fo], fi)))
            
    #making an array containing the classes each of the documents belong to
    Y = []
    for folder_name in folders:
        folder_path = join(my_path, folder_name)
        num_of_files= len(listdir(folder_path))
        for i in range(num_of_files):
            Y.append(folder_name)
            
    #choose documents 
    # we start with the positive bags
    seed(datetime.now())
    pathnames = []

    # put all positive docs into one list and negative docs in another
    positive_docs = [pathname_list[idx] for idx, element in enumerate(pathname_list) if Y[idx] == group]
    negative_docs = [pathname_list[idx] for idx, element in enumerate(pathname_list) if Y[idx] != group]
    for i in range(num_positive_bags):
        path_bag = []
        # select one positive bag
        random_index = randint(0, len(positive_docs)-1)
        current_file = positive_docs[random_index]
        positive_docs.pop(random_index) # remove the file we just used, so we don't use it again       
        path_bag.append(current_file)
        
        # select the rest of the bags, with positivity rate specified at the top of the function
        for j in range(num_instances - 1):
            if(random() < positivity_rate): # insert positive instance
                random_index = randint(0, len(positive_docs)-1)
                current_file = positive_docs[random_index]
                positive_docs.pop(random_index) # remove the file we just used, so we don't use it again         
                path_bag.append(current_file)
            else: # insert a negative instance
                random_index = randint(0, len(negative_docs)-1)
                current_file = negative_docs[random_index]
                negative_docs.pop(random_index) # remove the file we just used, so we don't use it again       
                path_bag.append(current_file)
        pathnames.append(path_bag[:])
                
    # create negative bags
    for i in range(num_bags - num_positive_bags):
        path_bag = []
        for j in range(num_instances):
            random_index = randint(0, len(negative_docs)-1)
            current_file = negative_docs[random_index]
            negative_docs.pop(random_index) # remove the file we just used, so we don't use it again       
            path_bag.append(current_file)
        pathnames.append(path_bag[:])
    
    #create vocabulary
    vocab = create_vocabulary_bow(num_features, pathnames)
    
    #create bags
    bags = []
    bag = []
    instance = []
    for paths in pathnames: # for each bag of paths in pathnames
        bag = []
        for path in paths: # for each path in the bag of paths
            instance = make_features_bow(vocab, path)
            bag.append(instance[:])
        bags.append(bag[:])
        
    # Create Labels
    labels = []
    for i in range(num_positive_bags):
        labels.append(1)
    for j in range(num_bags - num_positive_bags):
        labels.append(-1)
        
    return bags, labels, pathnames, vocab

def parse_20newsgroups_tfidf(): 
    # set parameters
    num_bags = 100
    num_positive_bags = 50
    num_instances = 50 
    num_features = 200 
    positivity_rate = 0.03

    group = input("Choose a group: enter the name of the group - ex: alt.atheism\n")
    
    my_path = '20_newsgroups'
    
    #creating a list of folder names to make valid pathnames later
    folders = [f for f in listdir(my_path)]
    
    
    #creating a 2D list to store list of all files in different folders
    files = []
    for folder_name in folders:
        folder_path = join(my_path, folder_name)
        files.append([f for f in listdir(folder_path)])
    
    
    #creating a list of pathnames of all the documents
    #this would serve to split our dataset into train & test later without any bias
    pathname_list = []
    for fo in range(len(folders)):
        for fi in files[fo]:
            pathname_list.append(join(my_path, join(folders[fo], fi)))
            
    #making an array containing the classes each of the documents belong to
    Y = []
    for folder_name in folders:
        folder_path = join(my_path, folder_name)
        num_of_files= len(listdir(folder_path))
        for i in range(num_of_files):
            Y.append(folder_name)
            
    #choose documents 
    # we start with the positive bags
    seed(datetime.now())
    pathnames = []
    # put all positive docs into one list and negative docs in another
    positive_docs = [pathname_list[idx] for idx, element in enumerate(pathname_list) if Y[idx] == group]
    negative_docs = [pathname_list[idx] for idx, element in enumerate(pathname_list) if Y[idx] != group]
    for i in range(num_positive_bags):
        path_bag = []
        # select one positive bag
        random_index = randint(0, len(positive_docs)-1)
        current_file = positive_docs[random_index]
        positive_docs.pop(random_index) # remove the file we just used, so we don't use it again       
        path_bag.append(current_file)
        
        # select the rest of the bags, with positivity rate specified at the top of the function
        for j in range(num_instances - 1):
            if(random() < positivity_rate): # insert positive instance
                random_index = randint(0, len(positive_docs)-1)
                current_file = positive_docs[random_index]
                positive_docs.pop(random_index) # remove the file we just used, so we don't use it again         
                path_bag.append(current_file)
            else: # insert a negative instance
                random_index = randint(0, len(negative_docs)-1)
                current_file = negative_docs[random_index]
                negative_docs.pop(random_index) # remove the file we just used, so we don't use it again       
                path_bag.append(current_file)
        pathnames.append(path_bag[:])
                
    # create negative bags
    for i in range(num_bags - num_positive_bags):
        path_bag = []
        for j in range(num_instances):
            random_index = randint(0, len(negative_docs)-1)
            current_file = negative_docs[random_index]
            negative_docs.pop(random_index) # remove the file we just used, so we don't use it again       
            path_bag.append(current_file)
        pathnames.append(path_bag[:])
    
    #create vocabulary
    vocab, idfDict = create_vocabulary_tfidf(num_features, pathnames)
    
    #create bags
    bags = []
    bag = []
    instance = []
    for paths in pathnames: # for each bag of paths in pathnames
        bag = []
        for path in paths: # for each path in the bag of paths
            instance = make_features_tfidf(vocab, path, idfDict)
            bag.append(instance[:])
        bags.append(bag[:])
        
    # Create Labels
    labels = []
    for i in range(num_positive_bags):
        labels.append(1)
    for j in range(num_bags - num_positive_bags):
        labels.append(-1)
        
    return bags, labels, pathnames, vocab


def parse_20newsgroups_doc2vec(): 
    # set parameters
    num_bags = 100
    num_positive_bags = 50
    num_instances = 50 
    num_features = 200 
    positivity_rate = 0.03
    
    group = input("Choose a group: enter the name of the group - ex: alt.atheism\n")

    my_path = '20_newsgroups'
    
    #creating a list of folder names to make valid pathnames later
    folders = [f for f in listdir(my_path)]
    
    
    #creating a 2D list to store list of all files in different folders
    files = []
    for folder_name in folders:
        folder_path = join(my_path, folder_name)
        files.append([f for f in listdir(folder_path)])
    
    
    #creating a list of pathnames of all the documents
    #this would serve to split our dataset into train & test later without any bias
    pathname_list = []
    for fo in range(len(folders)):
        for fi in files[fo]:
            pathname_list.append(join(my_path, join(folders[fo], fi)))
            
    #making an array containing the classes each of the documents belong to
    Y = []
    for folder_name in folders:
        folder_path = join(my_path, folder_name)
        num_of_files= len(listdir(folder_path))
        for i in range(num_of_files):
            Y.append(folder_name)
            
    #choose documents 
    # we start with the positive bags
    seed(datetime.now())
    pathnames = []
    # put all positive docs into one list and negative docs in another
    positive_docs = [pathname_list[idx] for idx, element in enumerate(pathname_list) if Y[idx] == group]
    negative_docs = [pathname_list[idx] for idx, element in enumerate(pathname_list) if Y[idx] != group]
    for i in range(num_positive_bags):
        path_bag = []
        # select one positive bag
        random_index = randint(0, len(positive_docs)-1)
        current_file = positive_docs[random_index]
        positive_docs.pop(random_index) # remove the file we just used, so we don't use it again       
        path_bag.append(current_file)
        
        # select the rest of the bags, with positivity rate specified at the top of the function
        for j in range(num_instances - 1):
            if(random() < positivity_rate): # insert positive instance
                random_index = randint(0, len(positive_docs)-1)
                current_file = positive_docs[random_index]
                positive_docs.pop(random_index) # remove the file we just used, so we don't use it again         
                path_bag.append(current_file)
            else: # insert a negative instance
                random_index = randint(0, len(negative_docs)-1)
                current_file = negative_docs[random_index]
                negative_docs.pop(random_index) # remove the file we just used, so we don't use it again       
                path_bag.append(current_file)
        pathnames.append(path_bag[:])
                
    # create negative bags
    for i in range(num_bags - num_positive_bags):
        path_bag = []
        for j in range(num_instances):
            random_index = randint(0, len(negative_docs)-1)
            current_file = negative_docs[random_index]
            negative_docs.pop(random_index) # remove the file we just used, so we don't use it again       
            path_bag.append(current_file)
        pathnames.append(path_bag[:])
    
    #create vocabulary
    vocab = create_vocab_doc2vec(num_features, pathnames)
    
    model = gensim.models.doc2vec.Doc2Vec(vector_size=num_features, min_count=2, epochs=1)
    
    model.build_vocab(vocab)
    
    model.train(vocab, total_examples=model.corpus_count, epochs=model.epochs)
    
    #create bags
    bags = []
    bag = []
    instance = []
    for paths in pathnames: # for each bag of paths in pathnames
        bag = []
        for path in paths: # for each path in the bag of paths
            instance = make_features_doc2vec(vocab, path, model).tolist()
            bag.append(instance[:])
        bags.append(bag[:])
        
    # Create Labels
    labels = []
    for i in range(num_positive_bags):
        labels.append(1)
    for j in range(num_bags - num_positive_bags):
        labels.append(-1)

    return bags, labels, pathnames, vocab

def create_vocab_doc2vec(num_features, pathnames):
    i = -1
    for paths in pathnames:
        for path in paths:
            words = []
            f = open(path, "r", encoding='latin1')
            text_lines = f.readlines()
            text_lines = remove_metadata(text_lines)
        
            #traverse over all the lines and tokenize each one with the help of helper function: tokenize_sentence
            for line in text_lines:
                words.append(tokenize_sentence(line))
                
            words = flatten(words)
            i += 1
            
            yield gensim.models.doc2vec.TaggedDocument(words, [i])

#make the features for the given pathname
def make_features_doc2vec(vocab, pathname, model):
    f = open(pathname, "r", encoding='latin1')
    
    text_lines = f.readlines()
    text_lines = remove_metadata(text_lines)
    
    doc_words = []
    
    #traverse over all the lines and tokenize each one with the help of helper function: tokenize_sentence
    for line in text_lines:
        doc_words.append(tokenize_sentence(line))
    
    # create features
    features = []
    
    features = model.infer_vector(flatten(doc_words))
    
    return features


def create_vocabulary_tfidf(num_features, pathnames):
    all_words = []
    num_documents = 0
    idfDict = {}
    for paths in pathnames:
        for path in paths:
            num_documents += 1
            f = open(path, "r", encoding='latin1')
            text_lines = f.readlines()
            text_lines = remove_metadata(text_lines)
        
            #traverse over all the lines and tokenize each one with the help of helper function: tokenize_sentence
            for line in text_lines:
                all_words.append(tokenize_sentence(line))
    
    # turn words into np array for further processing
    np_all_words = np.asarray(flatten(all_words))
    
    # find unique words and their frequency
    words, counts = np.unique(np_all_words, return_counts=True)
    
    #create dictionary to look up word counts
    wordCounts = dict(zip(words, counts))
    
    # sort words based off their frequency
    freq, wrds = (list(i) for i in zip(*(sorted(zip(counts, words), reverse=True))))
    
    # choose n number of top words
    vocab = wrds[0:num_features]
    
    # create bow
    bow = create_bow(pathnames)
    
    for word in vocab:
        num_contains = contains(bow, word)
        idfDict[word] = math.log(float(num_documents) / float(num_contains))
    
    return vocab, idfDict

#creates a bag of words for each document
def create_bow(pathnames):
    bows = []
    bow =[]
    for paths in pathnames:
        for path in paths:
            bow = []
            f = open(path, "r", encoding='latin1')
            text_lines = f.readlines()
            text_lines = remove_metadata(text_lines)
            for line in text_lines:
                bow.append(tokenize_sentence(line))
            bows.append(flatten(bow))
            
    return bows

# returns the number of documents that contain the word
def contains(bow, word):
    num_contains = 0
    for b in bow:
        for doc_word in b:
            if doc_word == word:
                num_contains += 1
                
    return num_contains

def make_features_tfidf(vocab, pathname, idfDict):
    f = open(pathname, "r", encoding='latin1')
    
    text_lines = f.readlines()
    text_lines = remove_metadata(text_lines)
    
    doc_words = []
    
    #traverse over all the lines and tokenize each one with the help of helper function: tokenize_sentence
    for line in text_lines:
        doc_words.append(tokenize_sentence(line))
    
    numWords = len(flatten(doc_words))
    
    # some documents have 0 words
    if(numWords == 0):
        return [0]*len(vocab)

    # turn words into np array for further processing
    np_doc_words = np.asarray(flatten(doc_words))
    
    # find unique words and their frequency
    words, counts = np.unique(np_doc_words, return_counts=True)    
    
    
    # create dictionary words -> counts
    dictionary = dict(zip(words,counts))
    
    # create features
    features = []
    
    for i in range(len(vocab)):
        tf = dictionary.get(vocab[i], 0)/numWords
        idf = idfDict.get(vocab[i])
        features.append(tf*idf)
    
    return features


def create_vocabulary_bow(num_features, pathnames):
    all_words = []
    for paths in pathnames:
        for path in paths:
            f = open(path, "r", encoding='latin1')
            text_lines = f.readlines()
            text_lines = remove_metadata(text_lines)
        
            #traverse over all the lines and tokenize each one with the help of helper function: tokenize_sentence
            for line in text_lines:
                all_words.append(tokenize_sentence(line))
    
    # turn words into np array for further processing
    np_all_words = np.asarray(flatten(all_words))
    
    # find unique words and their frequency
    words, counts = np.unique(np_all_words, return_counts=True)
    
    # sort words based off their frequency
    freq, wrds = (list(i) for i in zip(*(sorted(zip(counts, words), reverse=True))))
    
    # choose n number of top words
    vocab = wrds[0:num_features]
    
    return vocab

def make_features_bow(vocab, pathname):
    f = open(pathname, "r", encoding='latin1')
    
    text_lines = f.readlines()
    text_lines = remove_metadata(text_lines)
    
    doc_words = []
    
    #traverse over all the lines and tokenize each one with the help of helper function: tokenize_sentence
    for line in text_lines:
        doc_words.append(tokenize_sentence(line))
        
    # turn words into np array for further processing
    np_doc_words = np.asarray(flatten(doc_words))
    
    # find unique words and their frequency
    words, counts = np.unique(np_doc_words, return_counts=True)    
    
    # create dictionary words -> counts
    dictionary = dict(zip(words,counts))
    
    # create features
    features = []
    for i in range(len(vocab)):
        features.append(dictionary.get(vocab[i], 0))
    
    return features

#function to remove metadata
def remove_metadata(lines):
    for i in range(len(lines)):
        if(lines[i] == '\n'):
            start = i+1
            break
    new_lines = lines[start:]
    return new_lines

#function to convert a sentence into list of words
def tokenize_sentence(line):
    words = line[0:len(line)-1].strip().split(" ")
    words = preprocess(words)
    words = remove_stopwords(words)
    
    return words

#function to preprocess the words list to remove punctuations

def preprocess(words):
    #we'll make use of python's translate function,that maps one set of characters to another
    #we create an empty mapping table, the third argument allows us to list all of the characters 
    #to remove during the translation process
    
    #first we will try to filter out some  unnecessary data like tabs
    table = str.maketrans('', '', '\t')
    words = [word.translate(table) for word in words]
    
    punctuations = (string.punctuation).replace("'", "") 
    # the character: ' appears in a lot of stopwords and changes meaning of words if removed
    #hence it is removed from the list of symbols that are to be discarded from the documents
    trans_table = str.maketrans('', '', punctuations)
    stripped_words = [word.translate(trans_table) for word in words]
    
    #some white spaces may be added to the list of words, due to the translate function & nature of our documents
    #we remove them below
    words = [str for str in stripped_words if str]
    
    #some words are quoted in the documents & as we have not removed ' to maintain the integrity of some stopwords
    #we try to unquote such words below
    p_words = []
    for word in words:
        if (word[0] and word[len(word)-1] == "'"):
            word = word[1:len(word)-1]
        elif(word[0] == "'"):
            word = word[1:len(word)]
        else:
            word = word
        p_words.append(word)
    
    words = p_words.copy()
        
    #we will also remove just-numeric strings as they do not have any significant meaning in text classification
    words = [word for word in words if not word.isdigit()]
    
    #we will also remove single character strings
    words = [word for word in words if not len(word) == 1]
    
    #after removal of so many characters it may happen that some strings have become blank, we remove those
    words = [str for str in words if str]
    
    #we also normalize the cases of our words
    words = [word.lower() for word in words]
    
    #we try to remove words with only 2 characters
    words = [word for word in words if len(word) > 2]
    
    return words

#function to remove stopwords
def remove_stopwords(words):
    stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
     'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
     'can', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
     'each', 'few', 'for', 'from', 'further', 
     'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's",
     'hers', 'herself', 'him', 'himself', 'his', 'how', "how's",
     'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',
     "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
     'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours' 'ourselves', 'out', 'over', 'own',
     'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 
     'than', 'that',"that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", 
     "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 
     'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
     "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's",'will', 'with', "won't", 'would', "wouldn't", 
     'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 
     'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', '1st', '2nd', '3rd',
     '4th', '5th', '6th', '7th', '8th', '9th', '10th']
    words = [word for word in words if not word in stopwords]
    return words

#a simple helper function to convert a 2D array to 1D, without using numpy
def flatten(list):
    new_list = []
    for i in list:
        for j in i:
            new_list.append(j)
    return new_list

"""

println("Choose a feature extraction method: ")
println("1 - BoW")
println("2 - TFIDF")
println("3 - Doc2Vec")
ask = readline()
if ask == "1"
    println("Loading Dataset...")
    X, y, paths, vocab = py"parse_20newsgroups_bow()"
elseif ask == "2"
    println("Loading Dataset...")
    X, y, paths, vocab = py"parse_20newsgroups_tfidf()"
elseif ask == "3"
    println("Loading Dataset...")
    X, y, paths, vocab = py"parse_20newsgroups_doc2vec()"
else
    error("Please give valid input (1, 2, or 3).")
end

X = convert(Array{Float64, 3}, X)

temp = DataFrame[]
#iterate over 100 bags
for i=1:100
	#take all elements in bag
	x = X[i, :, :]	

	#create dataframe from bag
	df = x |> DataFrame

	#push dataframe to temp
	push!(temp, deepcopy(df))
end

#turn into one large dataframe
concat = temp[1]
for i=2:100
	append!(concat, temp[i])
end

#standardize dataframe
X_standardizer = Standardizer()
Xstandardized = MLJBase.transform(fit!(machine(X_standardizer, concat)), concat)

#turn back into bags
Xs = DataFrame[]
startIndex = 1
endIndex = 50
for i in 1:100
	global startIndex
	global endIndex

	#extract those rows
	x = Xstandardized[startIndex:endIndex, :]
	
	#update start and end
	startIndex = startIndex + 50
	endIndex = endIndex + 50

	#push into Xs
	push!(Xs, deepcopy(x))
end

y = CategoricalArray([i for i in y])

ordered!(y, true)

print("Verbose output? (y/n): ")
ask = readline()
if ask == "y"
    verbosity=10
elseif ask == "n"
    verbosity=1
else
    error("Please give valid input (y/n).")
end

evals_and_models = []

measure = [confusion_matrix, accuracy, bacc]
cv = CV(nfolds=6, shuffle=true)

sil = SIL(verbose=false); pysil = PyMISVMClassifier(sil)
#build_sil!(pysil, missing, dataset)
sil_machine = machine(pysil, Xs, y)
sil_eval = evaluate!(sil_machine, resampling=cv, measure=measure, verbosity=1)
@show sil_eval
push!(evals_and_models, (sil_eval, "SIL"))

#misvm = miSVM(verbose=false); pymisvm = PyMISVMClassifier(misvm)
#build_miSVM!(pymisvm, missing, dataset)
#misvm_machine = machine(pymisvm, Xs, y)
#misvm_eval = evaluate!(misvm_machine, resampling=cv, measure=measure, verbosity=1)
#@show misvm_eval


MIsvm = MISVM(verbose=false); pyMIsvm = PyMISVMClassifier(MIsvm)
#build_MISVM!(pyMIsvm, missing, dataset)
MIsvm_machine = machine(pyMIsvm, Xs, y)
MIsvm_eval = evaluate!(MIsvm_machine, resampling=cv, measure=measure, verbosity=1)
@show MIsvm_eval
push!(evals_and_models, (MIsvm_eval, "MISVM"))

sbmil = sbMIL(verbose=false); pysbmil = PyMISVMClassifier(sbmil)
#build_sbmil!(pysbmil, missing, dataset)
sbmil_machine = machine(pysbmil, Xs, y)
sbmil_eval = evaluate!(sbmil_machine, resampling=cv, measure=measure, verbosity=1)
@show sbmil_eval
push!(evals_and_models, (sbmil_eval, "sbMIL"))

smil = sMIL(verbose=false); pysmil = PyMISVMClassifier(smil)
#build_smil!(pysmil, missing, dataset)
smil_machine = machine(pysmil, Xs, y)
smil_eval = evaluate!(smil_machine, resampling=cv, measure=measure, verbosity=1)
@show smil_eval
push!(evals_and_models, (smil_eval, "sMIL"))

nsk = NSK(verbose=false); pynsk = PyMISVMClassifier(nsk)
#build_nsk!(pynsk, missing, dataset)
nsk_machine = machine(pynsk, Xs, y)
nsk_eval = evaluate!(nsk_machine, resampling=cv, measure=measure, verbosity=1)
@show nsk_eval
push!(evals_and_models, (nsk_eval, "NSK"))

stk = STK(verbose=false); pystk = PyMISVMClassifier(stk)
#build_stk!(pystk, missing, dataset)
stk_machine = machine(pystk, Xs, y)
stk_eval = evaluate!(stk_machine, resampling=cv, measure=measure, verbosity=1)
@show stk_eval
push!(evals_and_models, (stk_eval, "STK"))

ours = pdMISVMClassifier(C=0.01, μ=1e-4, ρ=1.02)
ours_machine = machine(ours, Xs, y)
ours_eval = evaluate!(ours_machine, resampling=cv, measure=measure, verbosity=verbosity)
@show ours_eval
push!(evals_and_models, (ours_eval, "Ours"))

function mean_and_std(per_fold)
	remove_nan = deleteat!(per_fold, isnan.(per_fold))     
	return string(round(mean(remove_nan), digits=3)) * "\$\\pm\$" * string(round(std(remove_nan), digits=3)) 
end

#println("Results for: " * dataset) 
println("Model \t Accuracy \t Balanced Accuracy") 
for tuple in evals_and_models    
	i, j = tuple     
	print(j, "\t")    
	print(mean_and_std(i.per_fold[2]))     
	print("\t")     
	println(mean_and_std(i.per_fold[3])) 
end
