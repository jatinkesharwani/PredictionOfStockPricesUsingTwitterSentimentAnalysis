!pip install yfinance
!pip install treeinterpreter
!pip install sklearn
!pip install nltk
!pip install tweepy

import tweepy
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
     

import pandas as pd
import sys
import re
import string
import json
import os
     

import nltk
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
sentiment_i_a = SentimentIntensityAnalyzer()

from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *   


from sklearn.model_selection import train_test_split
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import svm
from sklearn.svm import SVR 

from sklearn.metrics import mean_squared_error
from math import sqrt



def getStockDetails(stockname,start_time,end_time):
  """
    Fetches and plots the stock details for a given stock symbol within a specified time range.

    Args:
        stockname (str): The stock symbol.
        start_time (str): The start date in the format 'YYYY-MM-DD'.
        end_time (str): The end date in the format 'YYYY-MM-DD'.
    """
  company = yf.Ticker(stockname)
  stockData = yf.download(stockname, start=start_time, end=end_time)
  plt.title('Time series chart of Closing stocks for ')
  plt.plot(stockData["Close"])
  plt.show()
  print("\n")
  stockData.to_csv('stockData_' + stockname + '.csv')

class TweetCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.punc_table = str.maketrans("", "", string.punctuation) # to remove punctuation from each word in tokenize

    def compound_word_split(self, compound_word):
        """
        Splits a compound word into individual words.

        Args:
            compound_word (str): The compound word.

        Returns:
            List[str]: A list of individual words.
        """
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', compound_word)
        return [m.group(0) for m in matches]

    def remove_non_ascii_chars(self, text):
        """
        Removes non-ASCII characters from a text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with non-ASCII characters removed.
        """
        return ''.join([w if ord(w) < 128 else ' ' for w in text])

    def remove_hyperlinks(self,text):
        """
        Removes hyperlinks from a text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with hyperlinks removed.
        """
        return ' '.join([w for w in text.split(' ')  if not 'http' in w])

    def get_cleaned_text(self, text):
        """
        Cleans and preprocesses a text.

        Args:
            text (str): The input text.

        Returns:
            str: The cleaned text.
        """
        
        cleaned_tweet = text.replace('\"','').replace('\'','').replace('-',' ')
        cleaned_tweet =  self.remove_non_ascii_chars(cleaned_tweet)
        if re.match(r'RT @[_A-Za-z0-9]+:',cleaned_tweet):
            cleaned_tweet = cleaned_tweet[cleaned_tweet.index(':')+2:]
        cleaned_tweet = self.remove_hyperlinks(cleaned_tweet)
        cleaned_tweet = cleaned_tweet.replace('#','HASHTAGSYMBOL').replace('@','ATSYMBOL') # to avoid being removed while removing punctuations
        tokens = [w.translate(self.punc_table) for w in word_tokenize(cleaned_tweet)] # remove punctuations and tokenize
        tokens = [nltk.WordNetLemmatizer().lemmatize(w) for w in tokens if not w.lower() in self.stop_words and len(w)>1] # remove stopwords and single length words
        cleaned_tweet = ' '.join(tokens)
        cleaned_tweet = cleaned_tweet.replace('HASHTAGSYMBOL','#').replace('ATSYMBOL','@')
        cleaned_tweet = cleaned_tweet
        return cleaned_tweet

    def clean_tweets(self, tweets, is_bytes = False):   
        """
        Cleans and preprocesses a list of tweets.

        Args:
            tweets (list): A list of tweets to be cleaned.
            is_bytes (bool, optional): Indicates whether the tweets are in bytes format. Defaults to False.

        Returns:
            list: A list of cleaned tweets.
        """
        test_tweet_list = []
        for tweet in tweets:
            if is_bytes:
                test_tweet_list.append(self.get_cleaned_text(ast.literal_eval(tweet).decode("UTF-8")))
            else:
                test_tweet_list.append(self.get_cleaned_text(tweet))
        return test_tweet_list
    
    def clean_single_tweet(self, tweet, is_bytes = False):  
        """
        Cleans and preprocesses a single tweet.

        Args:
            tweet (str): The tweet to be cleaned.
            is_bytes (bool, optional): Indicates whether the tweet is in bytes format. Defaults to False.

        Returns:
            str: The cleaned tweet.
        """

        if is_bytes:
             return self.get_cleaned_text(ast.literal_eval(tweet).decode("UTF-8"))
        return self.get_cleaned_text(tweet)
    
    def cleaned_file_creator(self, op_file_name, value1, value2):
        """
        Creates a cleaned file in CSV format.

        Args:
            op_file_name (str): The name of the output file.
            value1 (list): The first column values for the CSV file.
            value2 (list): The second column values for the CSV file.

        Returns:
            None
        """
        csvFile = open(op_file_name, 'w+')
        csvWriter = csv.writer(csvFile)
        for tweet in range(len(value1)):
            csvWriter.writerow([value1[tweet], value2[tweet]])
        csvFile.close()

def fetchTweets(stockname, start_time, end_time):
  """
  Fetches tweets related to a stock within a specified time range and saves them in a CSV file.

  Args:
      stockname (str): The name of the stock.
      start_time (datetime.date): The start date for fetching tweets.
      end_time (datetime.date): The end date for fetching tweets.

  Returns:
      None
  """
  cleanObj = TweetCleaner()
  consumer_key    = 'm3GO6jv3CVa15qygE4CuCyWNt'
  consumer_secret = 'Jvd3tsKwE7daD8IfAH8jFjItVLaOnLkj8b6JqtdPE7v91mhujf'
  access_token  = '1447854807604088838-pXYWIGy1DL0s1ZLoA7EArP0kWHWETS'  
  access_token_secret = 'vXoyHyBPX96juIPRNhRAIk0mUFE3YiASL9E4D2Z4fPqsu'

  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  api = tweepy.API(auth,wait_on_rate_limit=True)
  fetch_tweets=tweepy.Cursor(api.search_tweets, q="#TSLA",count=100, lang ="en", tweet_mode="extended").items()
  start_date = datetime.date(2023, 3, 1)
  end_date = datetime.date(2023, 4, 1)


  delta = datetime.timedelta(days=1)
  curr_date = start_date

  tweets = []
  # while curr_date <= end_date:
  #   # Define search query
  #  since = curr_date.strftime('%Y-%m-%d')
  #  until = (curr_date + delta).strftime('%Y-%m-%d')
  #  query = f'{stockname} since:{since} until:{until}'

  #   # Retrieve one tweet for the current date
  #  try:
  #      for tweet in sntwitter.TwitterSearchScraper(query).get_items():
  #          tweets.append([tweet.date.strftime("%Y-%m-%d"), tweet.rawContent])
  #          print(tweet.content)
  #          break
  #  except:
  #      print('No tweets found')

  #   # Move to next date
  #  curr_date += delta
  for t in fetch_tweets:
   tweets.append([t.created_at.strftime("%Y-%m-%d"), t.full_text])


  # Write tweets to CSV file and dataframe
  with open(f'tweets_AAPL.csv', 'a', encoding='utf-8') as f:
    csvWriter = csv.writer(f, lineterminator='\n')
    csvWriter.writerow(['Date', 'Tweets'])
    for tweet in tweets:
      tweet_text = tweet[1].encode('utf-8')
      tweet_text = cleanObj.get_cleaned_text(tweet_text.decode())
      csvWriter.writerow([tweet[0], tweet_text])
      # print(tweet[1]+ "\n"+tweet_text)

  # Read CSV file into a dataframe
  # df = pd.read_csv(f'tweets_{stockname}.csv', names=['Date', 'Tweets'])
  # df.to_csv(f'tweets_{stockname}.csv', index=False, header=False, mode='a')
  # return df



def processTweets(stockname):
  """
    Process the tweets related to a stock by grouping them based on date and associating respective prices using stock data.

    Args:
        stockname (str): The name of the stock.

    Returns:
        None
  """
  columns=['Date','Tweets']
  data = pd.DataFrame(columns=columns)  # <- fix column parameter
  df = pd.read_csv('tweets_' + stockname  + '.csv', encoding='utf-8', names=columns, header=None)
  indx=0
  get_tweet=""
  #get tweets day wise
  for i in range(0,len(df)-1):
    get_date=df["Date"].iloc[i]
    next_date=df["Date"].iloc[i+1]
    if(str(get_date)==str(next_date)):
      get_tweet = get_tweet + df["Tweets"].iloc[i]+" "
    if(str(get_date)!=str(next_date)):
      get_tweet = df["Tweets"].iloc[i]
      dataf={'Date':get_date,'Tweets':get_tweet}
      # data['Date'].iloc[indx] = get_date
      # data['Tweets'].iloc[indx] = get_tweet
      # data.at[indx,'Tweets'] = get_tweet
      # data=data.concate(dataf, ignore_index=True)
      data = pd.concat([data, pd.DataFrame(dataf, index=[indx])], ignore_index=True)
      indx=indx+1
      get_tweet=" "

  #get respective prices for each day using stockData
  data['Prices']= np.nan
  readStockData = pd.read_csv('stockData_' + stockname + '.csv')
  readStockData.columns = [c.replace(' ', '_') for c in readStockData.columns]
  
  for i in range (0,len(data)):
      for j in range (0,len(readStockData)):
          get_tweet_date = data["Date"].iloc[i]
          get_stock_date = readStockData["Date"].iloc[j]
          # print(type(get_tweet_date) + " " + type(get_stock_date))
          if np.isin(get_tweet_date, get_stock_date):
            data["Prices"].iloc[i] = int(readStockData.Adj_Close[j])
            break
  

  data.dropna(subset=['Prices'], inplace=True)
  data.reset_index(drop=True, inplace=True)
  

  print(data.head())
  data.to_csv('processedTweets_' + stockname + '.csv')





def sentimentAnalysis(stockname):
  """
  Perform sentiment analysis on processed tweets related to a stock and generate a pie chart showing the distribution of sentiment.

  Args:
      stockname (str): The name of the stock.

  Returns:
      None
  """
  data = pd.read_csv('processedTweets_' + stockname  + '.csv', encoding='utf-8')
  data["Comp"] = ''
  data["Negative"] = ''
  data["Neutral"] = ''
  data["Positive"] = ''
  for indexx, row in data.T.iteritems():
    try:
      sentence_i = unicodedata.normalize('NFKD', data.loc[indexx, 'Tweets'])
      sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
      data.at[indexx, 'Comp'] =  sentence_sentiment['compound']
      data.at[indexx, 'Negative'] = sentence_sentiment['neg']
      data.at[indexx, 'Neutral'] =  sentence_sentiment['neu']
      data.at[indexx, 'Positive'] = sentence_sentiment['pos'] 
    except TypeError:
      print('failed on_status,',str(e))

  data.drop(['Unnamed: 0'], 1, inplace=True)
  print(data.head())
  data.to_csv('sentimentAnalysis_' + stockname  + '.csv')

  posi=0
  nega=0
  neutral = 0
  for i in range (0,len(data)):
    get_val = data.Comp[i]
    if(float(get_val)<(0)):
        nega=nega+1
    if(float(get_val>(0))):
        posi=posi+1
    if(float(get_val)==(0)):
        neutral=neutral+1
  
  posper=(posi/(len(data)))*100
  negper=(nega/(len(data)))*100
  neutralper=(neutral/(len(data)))*100

  arr=np.asarray([posper,negper,neutralper], dtype=int)
  plt.figure()
  plt.pie(arr,labels=['positive','negative', 'neutral'])
  plt.plot()

  print("% of positive tweets= ",posper)
  print("% of negative tweets= ",negper)
  print("% of neutral tweets= ",neutralper)

def RandomForestModel(stockname):
  """
  Train a Random Forest model on sentiment analysis data and make predictions for stock prices.

  Args:
      stockname (str): The name of the stock.

  Returns:
      None
  """
  df = pd.read_csv('sentimentAnalysis_' + stockname  + '.csv', encoding='utf-8')
  train, test = train_test_split(df, shuffle=False, test_size=0.2)

  sentiment_score_list_train = []
  for date, row in train.T.iteritems():
    sentiment_score = np.asarray([df.loc[date, 'Negative'],  df.loc[date, 'Neutral'], df.loc[date, 'Positive']])
    sentiment_score_list_train.append(sentiment_score)
  numpy_df_train = np.asarray(sentiment_score_list_train)

  sentiment_score_list_test = []
  for date, row in test.T.iteritems():
    sentiment_score = np.asarray([df.loc[date, 'Negative'],  df.loc[date, 'Neutral'], df.loc[date, 'Positive']])
    sentiment_score_list_test.append(sentiment_score)
  numpy_df_test = np.asarray(sentiment_score_list_test)

  y_train = pd.DataFrame(train['Prices'])
  y_test = pd.DataFrame(test['Prices'])

  rf = RandomForestRegressor()
  rf.fit(numpy_df_train, y_train)
  prediction, bias, contributions = ti.predict(rf, numpy_df_test)

  print("\n\n")
  plt.figure()
  plt.plot(test['Prices'].iloc[:].values)
  plt.plot(prediction.flatten())
  plt.title('Random Forest predicted prices')
  plt.ylabel('Stock Prices')
  plt.xlabel('Days')
  plt.legend(['actual', 'predicted'])
  plt.show()

  print("\n\n")
  print("RMSE value for Random Forest Model : ")
  rmse = sqrt(mean_squared_error(y_test, prediction.reshape(-1, 1)))
  print(rmse)
  print("\n\n")



def SVRModel(stockname):
  """
  Train a Support Vector Regression (SVR) model on sentiment analysis data and make predictions for stock prices.

  Args:
      stockname (str): The name of the stock.

  Returns:
      None
  """
  df = pd.read_csv('sentimentAnalysis_' + stockname  + '.csv', encoding='utf-8')
  train, test = train_test_split(df, shuffle=False, test_size=0.2)

  sentiment_score_list_train = []
  for date, row in train.T.iteritems():
    sentiment_score = np.asarray([df.loc[date, 'Negative'],  df.loc[date, 'Neutral'], df.loc[date, 'Positive']])
    sentiment_score_list_train.append(sentiment_score)
  numpy_df_train = np.asarray(sentiment_score_list_train)

  sentiment_score_list_test = []
  for date, row in test.T.iteritems():
    sentiment_score = np.asarray([df.loc[date, 'Negative'],  df.loc[date, 'Neutral'], df.loc[date, 'Positive']])
    sentiment_score_list_test.append(sentiment_score)
  numpy_df_test = np.asarray(sentiment_score_list_test)

  y_train = pd.DataFrame(train['Prices'])
  y_test = pd.DataFrame(test['Prices'])

  svr_rbf = SVR(kernel='rbf', C=1e6, gamma=0.1)
  svr_rbf.fit(numpy_df_train, y_train.values.flatten())
  output_test_svm = svr_rbf.predict(numpy_df_test)

  plt.figure()
  plt.plot(test['Prices'].iloc[:].values)
  plt.plot(output_test_svm)
  plt.title('SVM predicted prices')
  plt.ylabel('Stock Prices')
  plt.xlabel('Days')
  plt.legend(['actual', 'predicted'])
  plt.show()

  print("\n\n")
  print("RMSE value for Support Vector Regression Model : ")
  rmse = sqrt(mean_squared_error(y_test, output_test_svm))
  print(rmse)
  print("\n\n")

def main():
  """
  The main function for fetching tweets, processing them, performing sentiment analysis,
  and training predictive models for stock price prediction.

  Args:
      None

  Returns:
      None
  """
  # name = input("Enter a valid STOCKNAME of the Corporation: ") #enter the name of the company
  # start_date = input("Enter the Start Date in the following format[YYYY-MM-DD]: ") #enter the start date to fetch the tweets
  # end_date = input("Enter the End Date in the following format[YYYY-MM-DD]: " ) #enter the end date to fetch the tweets
  
  # if(len(name) > 0):
  #   STOCKNAME  = name
  # else:
  #   STOCKNAME = "AAPL"
  
  # if(len(start_date) > 0):
  #   start_time = start_date
  # else:
  #   start_time = "2018-01-01"
  
  # if(len(end_date) > 0):
  #   end_time = end_date
  # else:
  #   end_time = "2019-12-31"
  STOCKNAME = 'AAPL'
  start_time = '2023-05-01'
  end_time = '2023-05-18'

  # df = fetchTweets(STOCKNAME, start_time, end_time)
  # print(df)

  #Get Stock Details
  print("------------------------------ Getting Stock details -----------------------------")
  stockData = getStockDetails(STOCKNAME,start_time,end_time)
  print("Stock Details fetched! \n")

  #Fetching tweets
  print("------------------------------ Fetching Tweets -----------------------------")
  fetchTweets(STOCKNAME,start_time,end_time)
  print("Tweets fetched! \n")

  #Get tweets Per Day and get the stock closing values for each date
  print("------------------------------ Processing Tweets -----------------------------")
  processTweets(STOCKNAME)
  print("Processed Tweets ! \n")

  #Perform Sentiment Analysis
  print("------------------------------ Performing Sentiment Analysis -----------------------------")
  sentimentAnalysis(STOCKNAME)
  print("Completed Sentiment Analysis on Tweets ! \n\n")
  time.sleep(10);

  #Training and Predicting using Random Forest Regression Model
  print("--------  Training and Predicting using Random Forest Regression Model -------")
  RandomForestModel(STOCKNAME)
  print("\n \n")

  #Training and Predicting using Support Vecor Regression Model
  print("-------- Training and Predicting using Support Vector Regression Model ------------")
  SVRModel(STOCKNAME)
  print("\n \n")

main()