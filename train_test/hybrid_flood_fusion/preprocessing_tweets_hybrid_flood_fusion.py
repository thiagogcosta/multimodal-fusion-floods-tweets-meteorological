import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import TweetTokenizer
from hunspell import Hunspell
from collections import Counter
import pickle

class Preprocessing:
    
    def __init__(self, tweets, key_words, model, type_bow_tfidf, dic_hashtag):
        
        # textual-features folder location
        directory = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/textual-features/data/'
        
        # dictionaries folder location
        root_directory = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/textual-features/data/dicionarios/'

        self.matriz_tokens = []
        self.vec_ids = []
        self.vec_index = []
        self.vec_labels = []
        self.dic = Hunspell('pt_BR', hunspell_data_dir=root_directory)
        self.stopwords = set(nltk.corpus.stopwords.words('portuguese'))
        
        self.out_of_sentence = 0

        self.count_non_existent = 0
        self.count_corrected_accepted = 0
        self.count_corrected_non_accepted = 0

        self.vec_fasttext = []
        self.vec_notfasttext = []

        self.type_bow_tfidf = type_bow_tfidf
        
        arquivo = open(directory+'key_words_contextOK.txt', 'rb')
        dic_context = pickle.load(arquivo)# Read the stream from the file and rebuild the original object.
        arquivo.close() # close the file

        for count_tweets in range(len(tweets)):
            
            phrase = tweets.loc[count_tweets, 'text']
            id_str = tweets.loc[count_tweets, 'id_str']
            index = tweets.loc[count_tweets, 'index']
            related = tweets.loc[count_tweets, 'related']
            
            #---------------------------------CLEAN---------------------------------
            print(phrase)
            
            #*****************************REMOVAL OF LINKS***************************
            phrase = re.sub(r'(www|https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', phrase)

            #******************REMOVAL OF PROFILES, DUPLICATE CHARACTERS ETC******
            tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)

            clean_tweet = tweet_tokenizer.tokenize(phrase)

            #********************REMOVAL OF RANDOM HASHTAGS******************  

            tw_semhashtag = []

            for i in range(len(clean_tweet)):
                
                if (clean_tweet[i].find('#') != -1):
                    
                    for j in range(len(dic_hashtag)):
                        
                        hashtag = dic_hashtag.loc[j, 'hashtag']
                        correction = dic_hashtag.loc[j, 'correcao']

                        exist = 0

                        if (clean_tweet[i] == hashtag):

                            #print(tweet_limpo[i])
                            #print(hashtag)
                            #print('++++++++++++++++++++')
                            exist +=1
                            break
                    
                    if exist > 0:
                        tw_semhashtag.append(correction)
                else:
                    tw_semhashtag.append(clean_tweet[i])
            
            print('--------------')
            #print(tw_semhashtag)

            #*****************REMOVAL OF SPECIAL CHARACTERS, SUCH AS EMOTICONS ETC.*************

            tw_semEmoticons = []
            
            for i in tw_semhashtag:

                status = re.search(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ ]', i)

                if not status:

                    tw_semEmoticons.append(i)
            
            #print(tw_semEmoticons)
    
            #****************************STOP WORDS REMOVAL********************************
            
            tw_semstopword = []

            for i in tw_semEmoticons: 

                if i not in self.stopwords:
                    
                    tw_semstopword.append(i)

            #print(tw_semstopword)
            
            if self.type_bow_tfidf == False:
            
            #*******************FAST TEXT and DICTIONARY VERIFICATION***********************
                tw_fasttext = []

                for i in tw_semstopword:
                    try:
                        if(i in model.vocab):
                            tw_fasttext.append(i)
                            self.vec_fasttext.append(i)
                        
                        elif i in dic_context:

                            result = dic_context[i]
                            
                            #print('---------------------------')
                            #print('word: ', i)
                            #print('corrected word: ', result)

                            aux = result.split()

                            #self.vec_notfasttext.append(i)                                             
                            #not_fast = 1
                            #arquivo.writelines(['word: '+str(i), '\n'])
                            
                            #print('dictionary words: ',result)

                            for j in aux:

                                if(j in model.vocab):
                                    tw_fasttext.append(j)
                            
                            self.count_corrected_accepted +=1

                        elif len(self.dic.suggest(i)) > 0:
                            
                            #print('word: ',i)

                            #print('suggested words: ',self.dic.suggest(i))
                            #arquivo.writelines(['suggested words: '+str(self.dic.suggest(i)), '\n'])
                            count_exist = 0

                            for j in self.dic.suggest(i):
                                
                                if j in model.vocab:
                                    tw_fasttext.append(j)
                                    self.count_corrected_accepted+=1
                                    
                                    #print('suggested word: ',j)
                                    #arquivo.writelines(['suggested words: '+str(j), '\n'])
                                    count_exist +=1
                                    break

                            if count_exist == 0:
                                self.count_corrected_non_accepted +=1
                                
                                #print('words were not accepted!')
                                #arquivo.writelines(['words were not accepted!', '\n'])

                        else:
                            self.count_non_existent += 1

                            #print('non-existent word:',i)
                            #arquivo.writelines(['non-existent word: '+str(i), '\n']) 
                        
                        #print('#####################')
                        #arquivo.writelines(['-----------------------------', '\n'])
                    except ValueError as e:
                        print('Erro:', e)
                        pass
                
                #print(tw_fasttext)
                #print('-------------------------------')

            else:
                
                tw_fasttext = tw_semstopword

            size = len(tw_fasttext)
            
            if size != 0:

                self.matriz_tokens.append(tw_fasttext)
                self.vec_ids.append(id_str)
                self.vec_index.append(index)
                self.vec_labels.append(related)

                print('accepted corrections:', self.count_corrected_accepted)
                print('corrected not accepted:',self.count_corrected_non_accepted)
                print('non-existent:', self.count_non_existent)

        
