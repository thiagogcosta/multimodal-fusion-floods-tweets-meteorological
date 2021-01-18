from gensim.models import KeyedVectors
import numpy as np

class Embeddings:

    def __init__(self, sentences, model):

        #*********ATTRIBUTES*********
        self.model = model
        self.sentences = sentences

    def getWordVec(self):

        vector_tweets = []

        #Loop in sentences
        for tweet in self.sentences:

            vector_of_words = []
            count_words = 0

            #Loop in words
            num_item = 0
            
            while num_item < len(tweet):
            
                try:
                    if count_words == 0:
                        vector_of_words = self.model[tweet[num_item]]
                    else:
                        vector_of_words = np.add(vector_of_words, self.model[tweet[num_item]])
                    count_words += 1
                except:
                    pass
                
                num_item +=1

            vector_of_words = np.asarray(vector_of_words)

            vector_of_words = vector_of_words / count_words
            #print('Tweet', tweet)
            #print('WordVec', vector_of_words.shape)

            #I create a tweets vector, i.e. a wordvec matrix
            vector_tweets.append(vector_of_words)

            print(vector_tweets)
            
        return vector_tweets