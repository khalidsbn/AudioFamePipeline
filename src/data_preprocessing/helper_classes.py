import re
import nltk
import emoji
import unicodedata
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class DropNaN(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    return X.dropna(axis=0, how='any')
  
class TextProcessing(BaseEstimator, TransformerMixin):
  def __init__(self, column_name='playlist_name', feature='playlist'):
    self.column_name = column_name
    self.processed_column_name = f'processed_{column_name}'
    self.feature = feature

  def fit(self, X, y=None):
    return self

  def normalize_text(self, text):
    # if text is null or not a string, return empty string
    if pd.isna(text) or not isinstance(text, str):
      return ""
    text = text.lower()
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

  def clean_text(self, text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Replace emojis with their textual description
    text = emoji.replace_emoji(text, replace='')
    # Remove special characters (keep letters, numbers, and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Collabse multiple space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

  def tokenize_and_remove_stopwords(self, text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return ' '.join([token for token in tokens if token not in stop_words])

  def lemmatize_text(self, text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

  def transform(self, X):
    X = X.copy()
    X[self.processed_column_name] = X[self.column_name].apply(self.normalize_text)
    X[self.processed_column_name] = X[self.processed_column_name].apply(self.clean_text)
    X[self.processed_column_name] = X[self.processed_column_name].apply(self.tokenize_and_remove_stopwords)
    X[self.processed_column_name] = X[self.processed_column_name].apply(self.lemmatize_text)
    # Feature name prefix
    feature_prefix = self.feature
    X[f'{feature_prefix}_length'] = X[self.processed_column_name].apply(len)
    X[f'{feature_prefix}_word_count'] = X[self.processed_column_name].apply(lambda x: len(x.split()))
    X[f'{feature_prefix}_sentiment'] = X[self.processed_column_name].apply(lambda x: TextBlob(x).sentiment.polarity)
    return X.drop(self.column_name, axis=1)
  
class DateProcessing(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = X.copy()
    X["track_album_release_date"] = pd.to_datetime(X["track_album_release_date"], errors="coerce")
    X["track_album_release_year"] = X["track_album_release_date"].dt.year.astype("Int64")
    X["track_album_release_month"] = X["track_album_release_date"].dt.month.astype("Int64")
    X["track_album_release_day"] = X["track_album_release_date"].dt.day.astype("Int64")
    return X.drop('track_album_release_date', axis=1)
  
  # To Process PlayList features
class SubgenreExtractor(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X = X.copy()
    X['playlist_subgenre_first_word'] = X.playlist_subgenre.str.split(' ').str[0]
    X['playlist_subgenre_last_word'] = X.playlist_subgenre.str.split(' ').str[-1]
    return X

# Audio features
class MsToMinutes(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X['duration_min'] = (X['duration_ms'] / 60000).round(2)
    return X.drop(['duration_ms'], axis=1)
  
# Audiioi features
from sklearn.preprocessing import StandardScaler
class TempoScaler(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    sclaer = StandardScaler()
    X['tempo_scaled'] = sclaer.fit_transform(X[['tempo']])
    return X.drop(['tempo'], axis=1)
