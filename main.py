import json
from pprint import pprint
import spacy
from spacy.lang.en import English
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import gensim
import argparse
import numpy as np
import sys
import string
from nltk.corpus import words as eng_words
from gensim import corpora
from gensim.test.utils import datapath
import os.path 
import pickle
from random import shuffle 
import nltk

nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
words = set(nltk.corpus.words.words())
spacy.load('en')
parser = English()


##function for tokenizing the text
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

##function for lemmatizing the words
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

##Function for cleaning the text and obtaining the tokens
def prepare_text_for_lda(text):
    tokens = tokenize(text) ##Tokenize the text : EN
    tokens = [token for token in tokens if len(token) > 4] ##Removing the words that are very short
    tokens = [token for token in tokens if token not in en_stop] ##Removing the common words
    tokens = [token for token in tokens if token in words] ##Removing words that are not in EN
    tokens = [get_lemma(token) for token in tokens] ##Lemmatizing the words
    return set(tokens) ##Returning a set of tokens, to remove repetition of words

def main():
    parser = argparse.ArgumentParser(description="To classify documents to different topics",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--infile", "-i") ##contains the path to the corpus
    parser.add_argument("--outfile", "-o") ##contains the path to write the output
    parser.add_argument("--topicfile", "-t") ##contains the path to write the output
    parser.add_argument("--modelpath", "-m") ##contains the path to save/load the model
    parser.add_argument("--dictionaryname", "-d") ##contains the path to save/load the model
    parser.add_argument("--num_topics", "-n", type=int) ##set the number of topics
    parser.add_argument('--train', help='to train/test',action='store_true')
    args = parser.parse_args()
    NUM_TOPICS = args.num_topics
    
    ##Reading the documents
    data=[]
    with open(args.infile) as f:
        for line in f:
            data.append(json.loads(line))

    #Obtaining the tokens for all documents
    text_data=[]
    for line in data:
        #tokens=tokenize(line['text'])
        tokens=prepare_text_for_lda(line['text'])
        text_data.append(tokens)
    # For training the LDA model
    if args.train:
        print("Training model")
        dictionary = corpora.Dictionary(text_data)
        dictionary.save(args.dictionaryname)
        train_data=text_data
        shuffle(train_data)
        corpus = [dictionary.doc2bow(text) for text in train_data]
        with open('corpus.pkl','wb') as f:
            pickle.dump(corpus,f)
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=30,
                                                           iterations=400, alpha='auto', chunksize=200)
        # Save model to disk.
        ldamodel.save(args.modelpath)
    #For reloading a pre-trained LDA model along with the corpus
    else:
        print("Reloading the model")
        if not os.path.isfile(args.dictionaryname): 
            print("File not found")
            sys.exit(0)
        dictionary = gensim.corpora.Dictionary.load(args.dictionaryname)
        corpus = pickle.load(open('corpus.pkl', 'rb'))
        ldamodel = gensim.models.ldamodel.LdaModel.load(args.modelpath)
    topics=[]
    open(args.outfile, 'w').close()
    for i, text in enumerate(text_data):
        bow = dictionary.doc2bow(text)
        t = ldamodel.get_document_topics(bow,minimum_probability=0.0001)
        output={}
        output["_id"]=str(data[i]['_id'])
        output["topics"]=np.argsort(np.array(t)[:,1])[::-1][:5].tolist()
        with open(args.outfile,'r+') as f: ##writing the ID along with the list of topics
            if len(f.read()) == 0:
                f.write(json.dumps(output))
            else:
                f.write('\n' + json.dumps(output))
    open(args.topicfile, 'w').close()
for i in range(NUM_TOPICS): ##Writing the words for all topics
        topic_words={'topic_num':i,'words':np.array((ldamodel.show_topic(i, topn=10)))[:,0].tolist()}
        with open(args.topicfile,'r+') as f:
            if len(f.read()) == 0:
                f.write(json.dumps(topic_words))
            else:
                f.write('\n' + json.dumps(topic_words))

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
