import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

df_audio = pd.read_csv('../../data/audio_data_clean.csv')
df_playlist = pd.read_csv('../../data/playlist_data_clean.csv')
df_track = pd.read_csv('../../data/track_data_clean.csv')

class MergeTrackData(BaseEstimator, TransformerMixin):
    def __init__(self, df_playlist, df_track):
        self.df_playlist = df_playlist
        self.df_track = df_track

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Assume X is df_audio
        df_playlist_unique = self.df_playlist.drop_duplicates(subset='track_id')
        df_track_unique = self.df_track.drop_duplicates(subset='track_id')

        df_merged = (
            pd.merge(X, df_playlist_unique, on='track_id')
              .merge(df_track_unique, on='track_id')
        )
        return df_merged

class FixTrackPopularityColumn(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=['track_popularity_x', 'track_id', 'track_album_id', 'playlist_id'])
        X.rename(columns={'track_popularity_y': 'track_popularity'}, inplace=True)
        return X

merge_pipeline = Pipeline([
    ('merge_tracks', MergeTrackData(df_playlist, df_track)),
    ('fix_track_popularity', FixTrackPopularityColumn()),
])

df = merge_pipeline.fit_transform(df_audio)

num_cols = df.select_dtypes(include=['number']).columns.tolist()
str_cols = df.select_dtypes(include=['object']).columns.tolist()

from sklearn.preprocessing import OrdinalEncoder
class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ordinal_encoder = OrdinalEncoder()
        X[self.columns] = ordinal_encoder.fit_transform(X[self.columns])
        return X
    
df[str_cols] = Encoder(str_cols).fit_transform(df[str_cols])
