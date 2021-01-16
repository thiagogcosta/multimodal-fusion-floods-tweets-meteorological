import time
import pandas as pd
import datetime

# textual-features folder location

textual_location = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-meteorological/textual-features/data/'

# Initial execution time
start = time.time()

#-----------------------------KEY-WORDS----------------------------------------

columns = ['palavras-chave']

keywords = pd.DataFrame(columns = columns)

phenomenon_words = ["alagamento","alagado","alagada",
                      "alagando","alagou","alagar",
                      "chove","chova","chovia","chuva",
                      "chuvarada","chuvosa","chuvoso",
                      "chuvona","chuvinha","chuvisco",
                      "chuvisqueiro","chuvendo","diluvio",
                      "dilúvio","enchente", "enxurrada",
                      "garoa","inundação","inundacao",
                      "inundada","inundado","inundar",
                      "inundam","inundou","temporal",
                      "temporais","tromba d'água"]

for i in phenomenon_words:
    
    index = len(keywords)
    
    keywords.loc[index, 'palavras-chave'] = i

#---------------------------------TWEETS NOV 2016 UNTIL NOV 2018---------------------------------

tweets_nov2016_nov2018 = pd.read_csv(textual_location + 'tweets_NOV2016_NOV2018.csv')

print(tweets_nov2016_nov2018)

tweets_nov2016_nov2018[['data', 'hora']] = tweets_nov2016_nov2018.created_at.str.split(' ',expand=True)

tweets_nov2016_nov2018['data'] = tweets_nov2016_nov2018.apply(lambda x: datetime.strptime(x['data'], '%Y-%m-%d'), axis=1)

minimum_limit = datetime.strptime('2016-11-07', '%Y-%m-%d')
maximum_limit = datetime.strptime('2018-11-07', '%Y-%m-%d')

tweets_nov2016_nov2018 = tweets_nov2016_nov2018[(tweets_nov2016_nov2018['data'] >= minimum_limit) & (tweets_nov2016_nov2018['data'] <= maximum_limit)]

tweets_nov2016_nov2018.reset_index(drop=True, inplace=True)

tweets_nov2016_nov2018.to_csv(textual_location + 'tweets_NOV2016_NOV2018_CERTO.csv')

print(tweets_nov2016_nov2018)

#---------------------------------TWEETS MAR 2016 UNTIL NOV 2018 - MORE KEYWORDS---------------------------------

columns = ['id_str', 'text', 'created_at', 'lon', 'lat', 'data', 'hora']

df_keywords = pd.DataFrame(columns = columns)

tweets_ok = pd.read_csv(textual_location + 'tweets_NOV2016_NOV2018_CERTO.csv')

vec = []

for j in range(len(keywords)):

    print('Keywords: ', keywords.loc[j]['palavras-chave'])
    aux = tweets_ok[tweets_ok['text'].str.contains(keywords.loc[j]['palavras-chave'])]

    print(len(aux))

    print('-------------------------------')

    vec.append(aux)

df_keywords = pd.concat(vec)

df_keywords.reset_index(drop=True, inplace=True)

print(df_keywords)

df_keywords.to_csv(textual_location + 'tweets_NOV2016_NOV2018_CERTO_keywords.csv')
