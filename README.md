# Topic-Modeling
This repository contains the codes for unspuervised document classification. The code takes in the documents along with their IDs and returns the Document ID with the corresponding topic IDs. It also generates the list of words under each topic. The following dependencies are to be installed before running the script. 
1. spacy==2.0.16
2. nltk==3.2.5
3. gensim==3.4.0

The code may work with different versions of the above mentioned packages, but the functionalities might differ. 

The main script has a number of arguments to be passed.
1. -i contains the link for the input file
2. -o contains the path for the output file which will contain the topic list
3. -t contains the path to the file where the words under each topic will be stored
4. -d contains the path to gensim dictionary (new dict will be created when training, old dictionary will be loaded if train is set to false)
5. -n specifies the number of topics that the documents must be categorised into
6. -m contains the link for saving the new model or loading a pretrained model
7. --train specifies whether to train a new model or to predict new documents based on a pre-trained model
