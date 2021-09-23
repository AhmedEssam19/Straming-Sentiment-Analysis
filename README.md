# Architecture
* We create a TCP socket between Twitter’s API and Spark, which waits for the call of the Spark Structured Streaming and then sends the Twitter data.
* We receive the data from the TCP socket and preprocess it with the pyspark library, which is Python’s API for Spark.
* We pass the data to RESTful API where the model is located.
* We apply sentiment analysis using fine-tuned BERT-Base-Uncased and return Positive, Negative, or Neutral.
* Finally, we save the tweet and the sentiment analysis polarity in a parquet file.

# Steps to Run The Application
* Run `pip install -r requirements.py` command.
* Download the saved model from this [link](https://drive.google.com/file/d/1Hu0_FrVG-F-sSG7x_3VSjqlae2FwcLf_/view?usp=sharing) into the same path of *.py files (`03_Scripts` directory).
* Go to `03_Scripts` directory.
* Run `chmod u+x run_server` command.
* Run `./run_server` command.
* Type the keyword you want in the ‘keyword.txt’ file.
* In another terminal, run `python twitter_connection.py`.
* In another terminal, run `python sentiment_analysis.py`.
* After a reasonable amount of time, terminate the running of `twitter_connection.py` and `sentiment_analysis.py`.
* In the end run, `get_insights.py` to get some insights about the keyword like how many tweets included that keyword and if the public is positive, negative, or neutral
about it.

# Evaluation
* Test set (unseen data): the model achieved 86.8% accuracy on the test set and 86.3%
on the validation set.
* Streaming data (keyword: Harry Potter)
* 
![alt text](https://github.com/AhmedEssam19/Straming-Sentiment-Analysis/blob/master/Pics/Screenshot%20from%202021-09-18%2019-24-48.png)
* Streaming data (keyword: Interstellar)
* 
![alt text](https://github.com/AhmedEssam19/Straming-Sentiment-Analysis/blob/master/Pics/Screenshot%20from%202021-09-18%2020-02-42.png)


### Dataset Link: https://www.kaggle.com/kazanova/sentiment140

