An LDA model is used to cluster similar documents from a corpus. The output file contains the topics for each document in the
decreasing order of the weights. The words correspoding to each set of topics is also generated. We can modify the number of 
topics. The entire approach for pre-processing is mentioned step by step. 
1. The first step is to clean the text and tokenize. Spacy's english language parser is used to tokenize the text. 
2. The most common words and the words with length less than 4 are removed. 
3. The root words are obtained using lemmatizer. 
4. The words that are not from the English language are also removed. 
5. A set of these tokens are returned for each document to reduce the word redundancy. 

Once the text pre-processing is done, a dictionary of the corpus is created using **corpora.Dictionary(data)** using the gensim corpora function. The corpus is created using the generated dictionary by the command **dictionary.doc2bow(text)**. This creates the Bag of Words representation of the dictionary. This comprises of creating the corpus using from the input documents. 

After this step is to train the latent Dirichlet allocation (LDA) model. This is a generative model that clusters the documents based on the similarity of the words in them. Once the model is trained, the next step is to use the trained model to assign topic IDs for each document. This is done using the command **ldamodel.get_document_topics(corpus, minimum_probability=0.0001)** where corpus is obtained using **dictionary.doc2bow(text)**. The output will contain the topics along with their probabilities. The words under each topic can be obtained using **ldamodel.show_topic**. This is the process for unsupervised clustering of documents.

