import requests
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql import functions as f


def preprocessing(lines):
    lines = lines.select(f.explode(f.split(lines.value, "t_end")).alias("word"))
    lines = lines.na.replace('', None)
    lines = lines.na.drop()
    lines = lines.withColumn('word', f.regexp_replace('word', r"(?:\@|https?\://)\S+", ''))
    lines = lines.withColumn('word', f.regexp_replace('word', 'RT', ''))
    lines = lines.withColumn('word', f.regexp_replace('word', r"[^a-zA-Z]", ' '))
    lines = lines.withColumn('word', f.lower(f.col('word')))

    return lines


def polarity_detection(text):
    resp = requests.post('http://127.0.0.1:5000/predict_polarity', data={'text': text})
    return resp.json()['prediction']


def text_classification(lines):
    # polarity detection
    polarity_detection_udf = f.udf(polarity_detection, StringType())
    lines = lines.withColumn("polarity", polarity_detection_udf("word"))
    return lines


if __name__ == "__main__":
    # get th keyword
    with open('keyword.txt') as file:
        keyword = file.read().strip()
    # create Spark session
    spark = SparkSession.builder.appName("TwitterSentimentAnalysis").getOrCreate()
    # read the tweet data from socket
    words = spark.readStream.format("socket").option("host", "0.0.0.0").option("port", 5557).load()
    # Preprocess the data
    words = preprocessing(words)
    # text classification to define polarity and subjectivity
    words = text_classification(words)
    words = words.repartition(1)
    query = words.writeStream.queryName("all_tweets").outputMode("append").format("parquet").\
        option("path", f"./parc_{keyword}").option("checkpointLocation", "./check").\
        trigger(processingTime='60 seconds').start()
    query.awaitTermination()
