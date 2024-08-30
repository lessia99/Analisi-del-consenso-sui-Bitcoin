# Databricks notebook source
# MAGIC %md
# MAGIC ### Analisi del consenso sul Bitcoin

# COMMAND ----------

# MAGIC %md
# MAGIC Carico le librerie necessarie

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, udf, year, avg, count, sum
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, FloatType, DoubleType
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

# COMMAND ----------

# MAGIC %md
# MAGIC Inizializzo una sezione di Spark

# COMMAND ----------

spark = SparkSession.builder \
    .appName("Bitcoin Sentiment Analysis") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC Carico il dataset

# COMMAND ----------

!wget https://proai-datasets.s3.eu-west-3.amazonaws.com/bitcoin_tweets.csv

import pandas as pd
dataset = pd.read_csv('/databricks/driver/bitcoin_tweets.csv')

tweets_df = spark.createDataFrame(dataset)
tweets_df.write.saveAsTable("bitcoin_tweets_1")

# COMMAND ----------

# MAGIC %md
# MAGIC Guardo com'è strutturato, numero di righe, di colonne, i tipi delle varie colonne

# COMMAND ----------

display(tweets_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Controllo eventuali valori nulli

# COMMAND ----------

# Calcolare il numero di valori nulli per ciascuna colonna
null_counts = tweets_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in tweets_df.columns])

null_counts.show()

# COMMAND ----------

# togliere la colonne url
tweets_df = tweets_df.drop('url')

tweets_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Per svolgere la sentiment analysis decido di usare VADER,e non Textblob, perchè è stato sviluppato con un focus su espressioni emotive e informali, comuni su piattaforme come Twitter e i social media inoltre utilizza un insieme di parole chiave e regole linguistiche per assegnare punteggi di sentiment (positivo, neutrale e negativo) ai testi. 
# MAGIC Nonostante Textblob offra un'analisi più accurata come la tokenizzazione e l'analisi grammaticale, VADER è più facile da implementare e può fornire risultati più rapidi per l'analisi di grandi volumi, inoltre è in grado di gestire automaticamente la tokenizzazione e la vettorizzazione del testo.

# COMMAND ----------

# MAGIC %md
# MAGIC Definisco una funzione per togliere le emoji

# COMMAND ----------

import emoji

def remove_emoji(text):
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  # emoticon
        u"\U0001F300-\U0001F5FF"  # simboli e pittogrammi
        u"\U0001F680-\U0001F6FF"  # trasporti e simboli mappa
        u"\U0001F1E0-\U0001F1FF"  # bandiere (i codici dei paesi)
        u"\U00002700-\U000027BF"  # vari simboli e frecce
        u"\U000024C2-\U0001F251" 
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

# COMMAND ----------

# MAGIC %md
# MAGIC Pulisco il testo 

# COMMAND ----------


# Funzione di pulizia del testo 
def clean_text(text):
    # rimuove URL e tag HTML
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)

    # rimuove menzioni di username
    text = re.sub(r'@\w+', '', text)

    # riuove hashtag ma mantiene il testo
    text = re.sub(r'#', '', text)

    # Rimuove emoji
    text = remove_emoji(text)

    # Rimuovepunteggiatura
    text = re.sub(r'[^\w\s]', '', text)

    # converte tutto in minuscolo
    text = text.lower()
    return text

clean_text_udf = udf(lambda x: clean_text(x), StringType())

tweets_df = tweets_df.withColumn('cleaned_text', clean_text_udf(tweets_df['text']))


# COMMAND ----------

display(tweets_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Con la pulizia del testo decido di tenere solo i caratteri latini, perdendo così alcuni twitter.

# COMMAND ----------

# MAGIC %md
# MAGIC ### VADER

# COMMAND ----------

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Funzione di analisi del sentiment con controllo per valori nulli
def get_vader_sentiment(text):
    if not isinstance(text, str):
        return 'neutral'
    sentiment = sid.polarity_scores(text)
    compound = sentiment['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

get_vader_sentiment_udf = udf(lambda x: get_vader_sentiment(x), StringType())

tweets_df = tweets_df.withColumn('sentiment', get_vader_sentiment_udf(tweets_df['cleaned_text']))

# COMMAND ----------

display(tweets_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Per riuscire a fare un grafico mappo i sentiment sui numeri

# COMMAND ----------

# Mappare i sentiment su numeri
sentiment_to_numeric = udf(lambda sentiment: 1.0 if sentiment == 'positive' else (-1.0 if sentiment == 'negative' else 0.0), FloatType())

tweets_df = tweets_df.withColumn('sentiment_numeric', sentiment_to_numeric(tweets_df['sentiment']))


# COMMAND ----------

# Utilizzare direttamente la colonna 'timestamp' per ottenere la data
tweets_df = tweets_df.withColumn('date', to_date(col('timestamp')))

# Filtrare i record con date valide
tweets_df = tweets_df.filter(col('date').isNotNull())

# Verificare i dati filtrati
tweets_df.select('timestamp', 'date').show()


# COMMAND ----------

# MAGIC %md
# MAGIC Controllo quanti record sono presenti per anno 

# COMMAND ----------

# Estrarre l'anno dalla colonna 'date'
tweets_df = tweets_df.withColumn('year', year(col('date')))

# Contare i record per ogni anno
yearly_counts = tweets_df.groupBy('year').agg(count('*').alias('record_count'))

# Ordinare i risultati per anno
yearly_counts = yearly_counts.orderBy('year')

# Convertire i risultati in Pandas DataFrame per facilitare la visualizzazione
yearly_counts_pd = yearly_counts.toPandas()

# Visualizzare i risultati
print(yearly_counts_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC Decido di rappresentare graficamente l'anno 2019, essendo quello con il maggior numero di tweet sui bitcoin a disposizione

# COMMAND ----------

# Filtrare i record per prendere solo i dati dal 2019 in poi
tweets_2019 = tweets_df.filter(year(col('date')) == 2019)

# Verificare i dati filtrati
tweets_2019.select('timestamp', 'date', 'sentiment', 'sentiment_numeric').show(10)


# COMMAND ----------

# calcolare la media dei sentiment per giorno
daily_sentiment = tweets_2019.groupby("date").agg(avg("sentiment_numeric").alias('average_sentiment'))

# COMMAND ----------

# Convertire i risultati in Pandas DataFrame per facilitare la visualizzazione
daily_sentiment_pd = daily_sentiment.toPandas()

# Grafico del sentiment giornaliero
plt.figure(figsize=(10, 5))
plt.bar(daily_sentiment_pd['date'], daily_sentiment_pd['average_sentiment'])
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.title('Daily Bitcoin Sentiment on Twitter with VADER')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Commento grafico 
# MAGIC Un attacco hacker a Binance (società che gestisce lo scambio delle criptovalute) nel 2019 è stato la causa dell'elevato numero di twitter registrati in quell'anno, decimo anniversario della nascita della cripto valuta, nonostante la grande oscillazione della valuta, dal grafico si può notare come i commenti positivi e neutrali siano in netta maggioranza rispetto a quelli negativi.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Risposte alle domande 

# COMMAND ----------

# MAGIC %md
# MAGIC Hanno più likes i commenti negativi o quelli positivi?

# COMMAND ----------

# Media dei likes per sentiment
likes_avg = tweets_df.groupBy('sentiment').agg(avg('likes').alias('average_likes'))

# Convertire in Pandas Dataframe per una migliore visualizzazione
likes_avg_pd = likes_avg.toPandas()

print(likes_avg_pd)

# COMMAND ----------

# Confrontare delle medie dei likes tra tweet negativi e positivi
negative_likes_avg = likes_avg_pd[likes_avg_pd['sentiment'] == 'negative']['average_likes'].values[0]
positive_likes_avg = likes_avg_pd[likes_avg_pd['sentiment'] == 'positive']['average_likes'].values[0]

if negative_likes_avg > positive_likes_avg:
    print("I tweet negativi hanno avuto più likes rispetto a quelli positivi.")
else:
    print("I tweet positivi hanno avuto più likes rispetto a quelli negativi.")

# COMMAND ----------

# MAGIC %md
# MAGIC I tweet negativi hanno avuto più interazioni (risposte) rispetto a quelli positivi?

# COMMAND ----------

# Media delle risposte per sentiment
replies_avg = tweets_df.groupBy('sentiment').agg(avg('replies').alias('average_replies'))

# Convertire in Pandas Dataframe per una migliore visualizzazione
replies_avg_pd = replies_avg.toPandas()

print(replies_avg_pd)

# COMMAND ----------

# Confrontare delle medie dei replies tra tweet negativi e positivi
negative_replies_avg = replies_avg_pd[replies_avg_pd['sentiment'] == 'negative']['average_replies'].values[0]
positive_replies_avg = replies_avg_pd[replies_avg_pd['sentiment'] == 'positive']['average_replies'].values[0]

if negative_replies_avg > positive_replies_avg:
    print("I tweet negativi hanno avuto più interazioni rispetto a quelli positivi.")
else:
    print("I tweet positivi hanno avuto più interazioni rispetto a quelli negativi.")
