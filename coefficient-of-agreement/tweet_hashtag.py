import pandas as pd
from nltk.tokenize import TweetTokenizer
import re

# coefficient-of-agreement folder location
local_coefficient = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/coefficient-of-agreement/data/'

tweets = local_coefficient + 'THIAGO_NOV2016_NOV2018_judge.csv'

tw = pd.read_csv(tweets)

#----------------------------------------------------

phenomenal_words = ["alagamento","alagado","alagada",
                      "alagando","alagou","alagar",
                      "chove","chova","chovia","chuva",
                      "chuvarada","chuvosa","chuvoso",
                      "chuvona","chuvinha","chuvisco","chuvendo",
                      "diluvio", "dilúvio", "enchente", "enxurrada",
                      "garoa","inundação","inundacao","inundada",
                      "inundado","inundar","inundam","inundou",
                      "temporal","temporais","tromba d'água"]

columns = ['hashtag', 'related']

dataframes = pd.DataFrame(columns = columns)

for i in range(len(tw)):
    
    phrase = tw.loc[i, 'text']
    related = tw.loc[i, 'related']
    
    #*****************************LINKS REMOVAL****************************
    phrase = re.sub(r'(www|https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', phrase)

    #print(phrase)

    #******************PROFILE REMOVAL, DUPLICATE CHARACTERS ETC******************
    tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)

    tweet_clean = tweet_tokenizer.tokenize(phrase)
    
    print(tweet_clean)
    
    print('-----------------------------------')
    
    for j in tweet_clean:
        
        if "#" in j:
            
            size = len(dataframes)
            dataframes.loc[size, 'hashtag'] = j
            
            count = 0
            
            for l in phenomenal_words:
                
                if l in j:
                    
                    count = 1
                    break
            
            if count == 1:
                dataframes.loc[size, 'word_fenomeno'] = 1
            else:
                dataframes.loc[size, 'word_fenomeno'] = 0

df_count = dataframes.groupby(['hashtag', 'word_fenomeno']).size().reset_index(name='counts')

df_count = df_count.sort_values('counts')

print(df_count)

df_count.to_csv(local_coefficient + 'THIAGO_NOV2016_NOV2018_judge_hashtags_rel.csv')
