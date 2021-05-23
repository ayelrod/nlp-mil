## Preface
This project was part of [Lodewijk Brand's](http://minds.mines.edu/people/lou/) (Lou) research project that I assisted with for a few months. I got this opportunity to assist Lou's work through the [Google Hidden Talents](https://cs.mines.edu/hiddentalents/) program at the Colorado School of Mines.

## About The Project
Lou is working on developing a machine learning model for Multiple-Instance Learning (MIL). You can learn more about MIL [here](https://en.wikipedia.org/wiki/Multiple_instance_learning). At the time of writing this, Lou's ML algorithm is not open source yet.

The tasks that I did mainly included parsing datasets to make them compatible with Lou's algorithm. I also got the chance to explore different feature extraction methods for natural language processing and I tried to find the best one for Lou's algorithm.

## Fox, Tiger, Elephant Dataset
My first task for Lou was to parse the Fox, Tiger, Elephant dataset into bags that could be read by his method. This was the first time I worked with multiple-instance learning and parsing datasets for it, so I was a bit lost at first but I picked it up quickly. Understanding the type of input that goes into MIL methods is not easy because it is multi-dimensional. Once I got that down, I wrote some parsing scripts for the [Fox](https://github.com/ayelrod/nlp-mil/blob/gh-pages/Fox%20Dataset.ipynb), [Tiger](https://github.com/ayelrod/nlp-mil/blob/gh-pages/Tiger%20Dataset.ipynb), and [Elephant](https://github.com/ayelrod/nlp-mil/blob/gh-pages/Elephant%20Dataset.ipynb) datasets.

This dataset is very interesting because it takes images and turns them into feature vectors (a list of numbers). If you are familiar with MIL, each bag in our dataset was an image, and each instance was an object within the image. The objects in the image were found using [Blobworld](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1999/5567.html). This is a problem that sets itself up very well for MIL because there are a different number of objects in each picture. The MIL method is trying to decide whether the animal of interest is found in that picture or not.

## Natural Language Processing
NLP is what I spent most of my time working on. At the time that I started this research project, I had no experience with natural language processing but I picked it up pretty quickly. After seeing an example of a feature vector in the fox, tiger, elephant dataset, it was apparent that I would have to find a way to turn a document into a list of numbers that could be read by an algorithm. Turning a document (essentially a list of words) into a feature vector (a list of numbers) is called feature extraction. There are many different ways to do this feature extraction, so my task was to find one that worked well with our MIL method.

### Parsing
I started by parsing the data to get a nice list of words out of each document. In this step, I had to remove metadata and [stop words](https://en.wikipedia.org/wiki/Stop_word) from the document to filter out unnecessary information. Then I created a feature vector out of the remaining list of words using three different methods, which I will explain below. I also chose the documents from the dataset that I was going to use in this step, in a very specific way that aligns with multiple-instance learning. 

### BoW
The first feature extraction method that I implemented was Bag of Words. This is a very simple feature extraction method that you can learn more about [here](https://en.wikipedia.org/wiki/Bag-of-words_model). Essentially, we create a vocabulary of words by choosing the 200 most frequent words in all of our selected documents. Then, our features become the number of times that each word appears in the document. There are other methods of choosing the vocabulary, but I chose to keep it simple.

### TF-IDF
The second feature extraction method that I implemented was TF-IDF. This method is very similar to BoW, but there is a few extra calculations done to get the final feature vector. You can learn more about TF-IDF [here](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). The vocabulary was chosen in the same way as BoW. Although this method is very similar to BoW, it performed a bit better in my testing. 

### Doc2Vec
The third and last feature extraction method that I implemented was Doc2Vec. This is the most complicated method, and you can learn more about it [here](https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e). Essentially, this is a feature extraction method based off of another famous method, Word2Vec. It gets quite complicated, but I implemented this using the [gensim](https://radimrehurek.com/gensim/models/doc2vec.html) library. This feature extraction method did not perform as well as I expected, unfortunately. 

### Code
The code for all of this is uploaded in the corresponding Github. All of the parsing and feature extraction code can be found [here](https://github.com/ayelrod/nlp-mil/blob/gh-pages/20news_test.jl). It is very messy because it was all copy and pasted from a jupyter notebook. I will also upload this notebook, and it can be found in the same repository. There is also a corresponding [README](https://github.com/ayelrod/nlp-mil/blob/gh-pages/20NewsREADME.md), but keep in mind this code is incomplete and is for demonstration purposes only. To actually run these methods, we would need access to Lou's code which is not open source at the moment. 

A lot of the helper functions used in parsing were taken from [here](https://github.com/gokriznastic/20-newsgroups_text-classification/blob/master/Multinomial%20Naive%20Bayes-%20BOW%20with%20TF.ipynb). I initially wrote these parsing functions in Python, but Lou uses Julia so I had to convert it. Luckily, there is a library called pycall, which allows me to do this without having to rewrite my code. That is the main reason for the code being messy and hard to read.

### Results
Once I had all my code set up, I began doing some testing. I found some interesting results that I wasn't necessarily impressed with. We were getting a balance accuracy of about 0.5 on most of the tests we ran. This might be concerning, but other established MIL algorithms performed the same or worse on the data. Our main goal was to prove that Lou's method was fast and as accurate as other methods. I think we achieved this goal.

My main focus was to find the best feature extraction method between BoW, TFIDF, and Doc2Vec. This was a long process of running the MIL methods repeatedly and trying to make sense of the results. I spent quite a bit of time trying to find the best tuning parameters for Lou's method, and I ended up finding some parameters that worked pretty well with the TFIDF extraction method. At that point, I began comparing the Lou's method to other MIL methods. You can read more about this and see the data in a short paper that I wrote up, which can be found [here](https://github.com/ayelrod/nlp-mil/blob/gh-pages/NLP_section.pdf). 

By the end of it, I found that TF-IDF worked the best. There was definitely some shortcomings in my experiments that I wrote about in my paper. However, I think I made some good progress on Lou's research and helped him understand his method better. I also learned a lot about myself and machine learning. Coming into this with no ML experience was daunting but I pushed myself and succeeded. I spent a lot of time learning, rather than just coding, but I proved to myself that I can do anything I want in the CS space. There wasn't much guidance when I was writing this code, so I had to learn it myself. I now feel confident trying new things and learning more on my own. I also learned a lot about machine learning, feature extraction, and my Python abilities. I even got to try a new programming language, Julia.

I want to thank Lou for mentoring me and answering all my questions. I also want to thank CS@Mines and Google for the opportunity to participate in this program. While the pandemic has been very rough, it has brought me great opportunities to work remotely with people that I otherwise would not have gotten the chance too. Thank you!
