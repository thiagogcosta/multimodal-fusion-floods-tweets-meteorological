# Abstract

Several occurrences of flooding in the city of São Paulo (Brazil) provide severe losses to the affected population, such as loss of material goods, contamination by water diseases, and even death. Thus, to mitigate the effect of flooding, Disaster Management aims to reduce the impact of flooding and assist affected people. The response stage of Disaster Management is even focused on identifying possible victims and developing rescue strategies. Therefore, to auxiliary the Disaster Management response stage, researchers from different parts of the world have often created computational mechanisms embedded with artificial intelligence to process social media information. This data can be considered virtual sensors of the population. However, the machines have difficulty interpreting textual information since they cannot identify the context in which the message was published. Therefore, in this work, we explore several types of computational mechanisms embedded with textual data and contextual information, besides proposing two models of Multimodal Fusion (such as Hybrid Flood Fusion and Hybrid Tweet Fusion) capable of obtaining the Situational Awareness of the flooding in the city of São Paulo from tweets, meteorological data and historical occurrences of flooding. Our results indicate that combining textual data with contextual information provides a more accurate obtaining of Situational Awareness of floods than unimodal strategies based on textual information. Methods for identifying flood areas based on empirical approaches to define group formation radius are more effective than those based on geostatistical guidelines to define group formation radius. Our approach to obtaining situational awareness of flooding from tweets and contextual information can be adapted to different languages, regions, and natural disasters.

# Setup

- Install Python 3.8.5;
    - For more information visit: https://www.python.org/downloads/.

- Install all modules contained in the file: requirements.txt;

- Install Mongo DB 3.6.8.
    - For more information visit: https://docs.mongodb.com/manual/administration/install-community/.

- Machine Settings:

    | Operational<br>System 	|               RAM Memory              	|                         CPU                        	|               Graphics Card              	|
    |:---------------------:	|:-------------------------------------:	|:--------------------------------------------------:	|:----------------------------------------:	|
    |      Ubuntu 19.10     	| 8GB Single<br>Channel<br>DDR3 1600MHz 	| 5th Generation<br>Intel Core i5-5200U<br>Processor 	| NVIDIA(R)<br>GeForce(R)<br>820M 2GB DDR3 	|

# Data access
- Get the textual data:

    - Register in the development section of the Twitter platform and create the credentials to have access to data published on the social network;
        - For more information visit: https://developer.twitter.com/en.
    - Create a crawler to download the shared messages on Twitter.
        - Period: November 7, 2016, to November 8, 2018;
        - The delimiting boxes: 
            - North: (-46.95, -23.62, -46.28, -23.33);
            - South: (-46.95, -23.91, -46.28, -23.62).

- Get the occurrences of flooding:
    
    - Manually capture the occurrences of flooding in São Paulo's city through the website of CGE-SP or prepare a crawler to capture it automatically;
        - Period: January 1, 2015, to October 30, 2018;
        - For more information visit: https://www.cgesp.org/v3/alagamentos.jsp.
    - Use google Geocoding API to convert the addresses of historical flood events to geographic coordinates.
        - For more information visit: https://developers.google.com/maps/documentation/geocoding/overview.

- Get the meteorological data:

    - Get the weather data of the city of São Paulo from the historical climate database.
        - Period: January 1, 2015 to October 30, 2018.
        - For more information visit: https://bdmep.inmet.gov.br/
        - For more information visit: https://portal.inmet.gov.br/dadoshistoricos

# How to run:
    
1. **coefficient-of-agreement** folder: 
    - Get the text data from the files: "**AYSLA_NOV2016_NOV2018_KEY**", "**LUAN_NOV2016_NOV2018_KEY**", "**THIAGO_NOV2016_NOV2018_KEY**";
        - The social network Twitter has a policy that makes it impossible to share personal information. Therefore, to obtain this information, use the Twitter Streaming API to capture the textual content from the tweets identifiers (**id_str**) contained in those files.
    - Execute the "**cont_tweets.py**" file to perform the joint evaluation of tweets related to flooding;
    - Execute the "**agreement.py**" file to perform the concordance check process between the judges;
    - Execute the "**tweet_hashtag.py**" file to get the most used hashtags in the selected tweets.

2. **textual-features** folder:
    - Get the text data from the files: "**tweets_NOV2016_NOV2018**";
        - The social network Twitter has a policy that makes it impossible to share personal information. Therefore, to obtain this information, use the Twitter Streaming API to capture the textual content from the tweets identifiers (**id_str**) contained in that file.
    - Create the directory called "**embeddings**" inside the "data" folder of the "**textual-features**" directory;
    - Create the directory called "**FastText**" inside the "**embeddings**" folder;
        - Insert the NILC FastText models of the SKIP-GRAM type with 50 and 100 dimensions and CBOW with 50 and 100 dimensions.
        - For more information: http://www.nilc.icmc.usp.br/embeddings.
    - Create the directory called "**Word2Vec**" inside the "**embeddings**" folder;
        - Insert the NILC Word2Vec models of the SKIP-GRAM type with 50 and 100 dimensions and CBOW with 50 and 100 dimensions.
        - For more information: http://www.nilc.icmc.usp.br/embeddings.
    - The folder called "dicionarios" means dictionaries, i.e., Brazilian Portuguese dicitionaries;
        - For more information: https://sites.google.com/site/latexgrucad/dicionario.
        - In addition those dictionaries is the same used by Libreoffice.
        - For more information: 
    - Execute the "**textual_processing.py**" to get all the messages of the Twitter that contain the hashtags related to flooding.

3. **meteorological-features** folder:
    - Execute the "**process_meteorological_features**" file to process the weather information of the city of São Paulo.
        - The execution of this step is optional, because it is possible to find the results of this step in the file: "**SP_A701_SAO_PAULO_MIRANTE_2015-2018-OK_ARRANGE_DATA_INTERPOLATE.csv**".

4. **flood-features** folder:
    - Get the address data from the files: "**ALAGAMENTOS-2018_2019-GEO_CERTO_INSIDE**" and "**alagamentos_EXTEND2018_2019_COORDS**" through google Geocoding API to convert the geographic coordinates to addresses of historical flood events.
        - This step is optional, because it is possible to find the results of the preprocessing of the historical flooding informations in the file: "**ALAGAMENTS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES.csv**".
    - Execute the file "**preprocessing_flood_climatic_features.py**" to preprocess historical occurrences of flooding;
        - The execution of this step is optional, because it is possible to find the results of this step in the file: "**ALAGAMENTS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES.csv**".
    - Execute the file "**group_flood_features.py**" to group historical occurrences of flooding by frequency and humidity;
    - Run the file "**flood_areas.py**" to run several experiments to identify areas prone to flooding events in the city of São Paulo, in addition to choosing the clustering approach that provides the most significant silhouette.

5. **ground-truth** folder:
    - Run the "**create_ground_truth.py**" file to process the heterogeneous information and combine the textual, meteorological, and geographic information. Also, elaborate the ground truth of Multimodal Fusion models;
    - Run the "**create-model-fusion.py**" file to create the training and testing files for the Multimodal Fusion models;

6. **train_test** folder:
    - - Get the tweet data from the files: "**THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_conjunto_verdade_OK**", "**THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_early_fusion**", and "**THIAGO_NOV2016_NOV2018_judge_CERTO_INSIDE_sp_cluster_FEATURES_modelo_late_fusion_tweet**";
        - The social network Twitter has a policy that makes it impossible to share personal information. Therefore, to obtain this information, use the Twitter Streaming API to capture the textual content from the tweets identifiers (**id_str**) contained in those files.
    - **early_fusion** folder:
        - Run the file "**create_training_tweets_early.py**" to preprocess the heterogeneous information, i.e., apply the cleaning and data transformation processes;
        - Run the file "**train_model_early_fusion.py**" to create the necessary files to train the Multimodal Fusion model from the previously preprocessed information, and train several machine learning algorithms to obtain an Early Multimodal Fusion model precisely;
        - Execute the file "**eval_ground_truth_early_fusion.py**" to create the necessary files to test the Multimodal Fusion model and test the Multimodal Fusion model of the previous type trained on a data set unknown to the classifier.
    - **hybrid_flood_fusion** folder:
        - Run the file "**train_model_hybrid_flood_fusion.py**" to create the necessary files to train the Multimodal Fusion model, and train several machine learning algorithms to obtain a Hybrid Flood Multimodal Fusion model precisely;
        - Run the file "**eval_ground_hybrid_flood_fusion.py**" to create the necessary files to test the Multimodal Fusion model and test the model called Hybrid Flood Fusion that focuses decisions on weather information and uses a flood identifier in a data set unknown to the Multimodal Fusion model.
    - **hybrid_tweet_fusion** folder:
        - Run the file "**train_model_hybrid_tweet_fusion.py**" to create the necessary files to train the Multimodal Fusion model, and train several machine learning algorithms to obtain a Hybrid Tweet Multimodal Fusion model precisely;
        - Run the file "**eval_ground_hybrid_tweet_fusion.py**" to create the necessary files to test the Multimodal Fusion model and test the model called Hybrid Tweet Fusion that focuses decisions on textual information and uses a tweet classifier in a data set unknown to the Multimodal Fusion model.
    - **late_fusion** folder:
        - Run the file "**create_training_tweets_late.py**" to preprocess the textual information, i.e., apply the cleaning and data transformation processes;
        - Run the file "**train_tweet_late_fusion.py**" to create the necessary files to train the Multimodal Fusion model from the previously preprocessed information, and train several machine learning algorithms to obtain a tweet classifier precisely;
        - Run the file "**train_flood_late_fusion.py**" train several machine learning algorithms to obtain a meteorological classifier precisely;
        - Run the "**eval_ground_truth_late_fusion.py**" file to create the files needed to test the Multimodal Fusion model, and test the Late Fusion model in a data set unknown to the classifier.
    - **unimodal_fusion** folder:
        - Run the file "**create_training_unimodal_fusion.py**" to preprocess the textual information, i.e., apply the cleaning and data transformation processes;
        - Run the file "**train_model_unimodal_fusion.py**" to create the necessary files to train the textual classifier model from the previously preprocessed information, and train several machine learning algorithms to obtain a Unimodal Fusion model precisely;
        - Execute the file "**eval_ground_truth_unimodal_fusion.py**" to create the necessary files to test a Unimodal Fusion model and test the textual classifier model of the previous type trained on a data set unknown to the classifier.

# Other important informations:

- The "**haversine.py**" file has the function of calculating the distance between two geographic coordinates taking into account the curvature of the planet earth.
- The "**inside_shapefile.py**" file has the function of identifying the geographic points that are contained in a specific region from the shapefile of the studied area.
- The "**semivariogram.r**" file has the function of calculating the stabilization distance of historical occurrences of flooding.
- The "**con_mongodb.py**" file has the function of providing functionalities to connect, add, change, and delete Mongo DB database collections.
- The "**embeddings.py**" file functions to transform the sentences of Twitter messages into word embeddings matrices.
- The files "**ml_early_fusion.py**", "**ml_hybrid_flood_fusion.py**", "**ml_hybrid_tweet_fusion.py**", "**ml_late_fusion.py**" and "**ml_unimodal_fusion.py**" provide the functionality of cross-validation, parameter optimization, as well as training and testing of Machine Learning algorithms.
- The files "**preprocessing_tweets_early_fusion.py**", "**preprocessing_tweets_hybrid_flood_fusion.py**", "**preprocessing_tweets_hybrid_tweet_fusion.py**", "**preprocessing_tweets_late_fusion.py**" and "**preprocessing_tweets_unimodal_fusion.py**" provide features capable of removing duplicate characters and e-mail addresses from tweets, as well as deleting special characters and stop words from Twitter messages. Furthermore, it checks and corrects the words contained in tweets for the Portuguese language cult norm.