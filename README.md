# Analisi-del-consenso-sui-Bitcoin

Un'azienda di ricerche di mercato deve stimare il consenso delle persone verso il Bitcoin. Un team di data engineer ha estratto diversi milioni di tweet che parlano di Bitcoin, l'obiettivo è quello di eseguire un'analisi del sentiment e creare un grafico che mostri come questo è variato giorno per giorno. 

## Data
1. **Dataset: bitcoin_tweets**: formato da 10 000 samples e 9 features:
   - *id*
   - *user*
   - *fullname*
   - *url*
   - *timestamp*
   - *replies*
   - *likes*
   - *retweets*
   - *text*

2. **Analisi sul consenso sul Bitcoin**: notebook contente l'analisi usando PySpark

## Dettagli analisi

Per svolgere la sentiment analysis decido di usare VADER,e non Textblob, perchè è stato sviluppato con un focus su espressioni emotive e informali, comuni su piattaforme come Twitter e i social media inoltre utilizza un insieme di parole chiave e regole linguistiche per assegnare punteggi di sentiment (positivo, neutrale e negativo) ai testi. 
Nonostante Textblob offra un'analisi più accurata come la tokenizzazione e l'analisi grammaticale, VADER è più facile da implementare e può fornire risultati più rapidi per l'analisi di grandi volumi, inoltre è in grado di gestire automaticamente la tokenizzazione e la vettorizzazione del testo.

1. **Pulizia del testo**
2. **VADER**
3. **Rappresentazione grafica**

## Prerequisiti e utilizzo

- Python
- Per questo progetto ho usato PySpark e i suoi vari strumenti, si può lavorare con PySpark in cloud gratuitamente utilizzando DataBricks Community
- importare:
  - nltk
  - matplotlib
  -  re
  -  pandas

Consiglio di aprire il file con estensione .py su Google Colab

- link dataset: https://proai-datasets.s3.eu-west-3.amazonaws.com/bitcoin_tweets.csv.

### Utilizzo
1.**Inizializzo una sezione di Spark**

spark = SparkSession.builder \  
    .appName("Bitcoin Sentiment Analysis") \  
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \  
    .getOrCreate()

2.**Caricamento del dataset**  
  Per poter caricare il dataframe e trasformarlo in una table basta eseguire su Notebook Databricks le seguenti righe di codice:
  
  !wget https://proai-datasets.s3.eu-west-3.amazonaws.com/bitcoin_tweets.csv
  import pandas as pd
  dataset = pd.read_csv('/databricks/driver/bitcoin_tweets.csv', delimiter=";")

  tweets_df = spark.createDataFrame(dataset)
  tweets_df.write.saveAsTable("bitcoin_tweets")

## Contatti
Per domande, suggerimenti o feedback, contattatemi pure alla mail alessiaagostini53@gmail.com.
