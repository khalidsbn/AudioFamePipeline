import pandas as pd
from sklearn.pipeline import Pipeline
from helper_classes import DropNaN, TextProcessing, DateProcessing

# Load your dataset
df = pd.read_csv('../../data/processed/train.csv')

# Group features
track_metadata = [
    'track_id', 'track_name', 'track_artist', 'track_popularity',
    'track_album_id', 'track_album_name', 'track_album_release_date'
]

playlist_metadata = [
     'track_id', 'playlist_name', 'playlist_id', 'playlist_genre', 'playlist_subgenre'
]

audio_features = [
    'track_id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'track_popularity'
]

# Create separate DataFrames for each group
df_track = df[track_metadata]
df_playlist = df[playlist_metadata]
df_audio = df[audio_features]

track_pipeline = Pipeline([
    ('TrackNameProcessing', TextProcessing(column_name='track_name', feature='track')),
    ('TrackArtistProcessing', TextProcessing(column_name='track_artist', feature='artist')),
    ('TrackAlbumNameProcessing', TextProcessing(column_name='track_album_name', feature='album')),
    ('DateProcessing', DateProcessing()),
    ('DropNaN', DropNaN())
])