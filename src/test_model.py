import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.data_preprocessing.helper_classes import (
    DropNaN, TextProcessing, DateProcessing, SubgenreExtractor,
    MsToMinutes, TempoScaler, FixTrackPopularityColumn,  Encoder,
    MergeTrackData
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    max_error
)

# Load your dataset
df = pd.read_csv('../data/processed/test.csv')

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

# Split the dataset into features and target variable
X = df.drop(columns=['track_popularity'])
y = df['track_popularity']

# Load the trained model
xgb = joblib.load('../models/xgb_model.pkl')

# Run the model on the test set
y_pred = xgb.predict(X)

# Evaluate the model
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
max_err = max_error(y, y_pred)

# Print Evaluation metrics
print("----- XGBoost Regressor Evaluation -----")
print(f"MAE  (Mean Absolute Error): {mae:.4f}")
print(f"MSE  (Mean Squared Error): {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"R²   (R-squared Score): {r2:.4f}")
print(f"Max Error: {max_err:.4f}")

# Display top 5 feature importances
importances = xgb.feature_importances_
feature_names = X.columns
importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\nTop 5 Important Features:")
print(importances_df.head(5))
