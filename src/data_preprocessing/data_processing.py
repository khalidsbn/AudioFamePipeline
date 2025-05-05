import pandas as pd
from sklearn.pipeline import Pipeline
from helper_classes import DropNaN, TextProcessing, DateProcessing, SubgenreExtractor
from helper_classes import MsToMinutes, TempoScaler # Audio featues
from helper_classes import MergeTrackData, FixTrackPopularityColumn, Encoder # To prepare data for training

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

# Pipelines for data processing
track_pipeline = Pipeline([
    ('TrackNameProcessing', TextProcessing(column_name='track_name', feature='track')),
    ('TrackArtistProcessing', TextProcessing(column_name='track_artist', feature='artist')),
    ('TrackAlbumNameProcessing', TextProcessing(column_name='track_album_name', feature='album')),
    ('DateProcessing', DateProcessing()),
    ('DropNaN', DropNaN())
])

playlist_pipeline = Pipeline([
    ('PlaylistProcessing', TextProcessing(column_name='playlist_name', feature='playlist')),
    ('Subgenre', SubgenreExtractor())
])

audio_features_pipeline = Pipeline([
    ('MsToMinutes', MsToMinutes()),
    ('TempoScaler', TempoScaler())
])

# Process the datasets
df_track = track_pipeline.fit_transform(df_track)
df_playlist = playlist_pipeline.fit_transform(df_playlist)
df_audio = audio_features_pipeline.fit_transform(df_audio)

# Save the processed DataFrames
df_track.to_csv('../../data/processed/track_data_clean.csv', index=False)
df_playlist.to_csv('../../data/processed/playlist_data_clean.csv', index=False)
df_audio.to_csv('../../data/processed/audio_data_clean.csv', index=False)

# Prepare datasets for training

# Merging Pipeline
merge_pipeline = Pipeline([
    ('merge_tracks', MergeTrackData(df_playlist, df_track)),
    ('fix_track_popularity', FixTrackPopularityColumn()),
])

# Merge dataset
df = merge_pipeline.fit_transform(df_audio)

# Select string features for encoding
str_cols = df.select_dtypes(include=['object']).columns.tolist()

# Encode string feature
df[str_cols] = Encoder(str_cols).fit_transform(df[str_cols])

# Save the final dataset
df.to_csv('../../data/features/data_cleaned.csv', index=False)
