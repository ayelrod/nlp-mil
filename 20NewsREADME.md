# 20news_test.jl

The 20news_test.jl file can be used to test multiple machine learning methods on the 20newsgroups dataset, including our pdMISVM classifier. It uses pyCall to call the python functions used to parse the 20newsgroup dataset into usable input. 

## Set up:
- Install all necessary packages listed at the top of the file. These packages are included in lines 1-34.
- Set path to wrappers folder and RepeatedCV.jl file (lines 11-12)

## Notes:
- There are some lines of code commented out because the current version of this script does not support these functionalities. Most notably, when calling different machine learning methods at the bottom of the file, the 'build' lines are commented out.

## Running The Script:
- Initializing everything takes a while, so give it some time before the script outputs anything.
- The first thing that you will be asked is which feature extraction method you would like to use. Choose either 1, 2, or 3. I will go over these below.
- The second thing that you will be asked is which newsgroup you want to use. The newsgroups can be found in the '20_newsgroups' folder.
- The script will create the dataset, then standardize the data.
- The next thing you will be asked concerns the verbosity of the output. For most purposes, enter 'n' here.
- The methods will start running and print out the results.

# The Dataset
The 20newsgroup dataset contains about 20,000 documents that are short news articles from different groups. The names of these groups are the folder names withing the '20_newsgroups' folder. The goal of our machine learning methods is, given a document, to correctly classify whether or not the document belongs to the group we are focusing on. For example, if we choose the 'rec.motorcycles' group, we first train our model and then it should be able to tell us, given a document, whether or not that document is about motorcycles. 

The default parameters are:
- 100 bags - 50 positive, 50 negative
- 50 documents/instances per bag
- 200 features per document
- The positivity rate for selecting documents that belong to the newsgroup we are focusing on is 0.03. This means that after we initially put one positive instance into our positive bag, we select documents at random with a 0.03 chance of selecting another positive document.

There are variables at the top of the "parse_*" python functions to change these parameters easily. However, some of the julia code towards the bottom of the file relies on these parameters being defualt, so you will also need to figure out which ones those are and change those if you want to change the parameters.

# Feature Extraction Methods
## BoW
Bag of words (BoW) is a very simple feature extraction method. First, we create a vocabulary of the 200 most frequent words in all the documents selected. The features for each document then become the number of times each word in the vocabulary appears.

## TFIDF
TFIDF is very similar to BoW, and we first create a vocabulary in the same way. However, instead of the frequency becoming the features, we do some further calculations on the words. More information can be found here: http://www.tfidf.com/

## Doc2Vec
Doc2Vec is a feature extraction method that builds off of Word2vec, a method developed by Google. In the 20news_test.jl file, Doc2Vec was implemented using the gensim library. More information can be found here: https://radimrehurek.com/gensim/models/doc2vec.html

* Some parsing code taken from: https://github.com/gokriznastic/20-newsgroups_text-classification/blob/master/Multinomial%20Naive%20Bayes-%20BOW%20with%20TF.ipynb
