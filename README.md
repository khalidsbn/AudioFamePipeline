<div align="center">
  <h1>Which Audio Features Most Significantly Influence a Track’s Popularity on Spotify?</h1>
</div>

# Project Overview

This repository contains an end-to-end machine learning pipeline designed to predict track popularity on Spotify. The project aims to analyze the relationship between various audio features and a track's popularity, with a focus on answering the following research question:

**“Which audio features most significantly influence a track’s popularity on Spotify?”**

The pipeline automates the data extraction, cleaning, and feature engineering process using Spotify's audio features, followed by the development of a regression model to predict track popularity. The project also includes an analysis of feature importance to identify the key attributes that drive track success.

## Key Features

- **Data Collection and Preprocessing**: Collects Spotify track data through the Spotify API, followed by preprocessing and cleaning to prepare the data for analysis.
- **Feature Engineering**: Identifies relevant audio features such as danceability, energy, tempo, and others that influence track popularity.
- **Model Development**: Trains a regression model to predict track popularity based on these features.
- **Feature Importance Analysis**: Implements techniques to assess which audio features have the most significant impact on a track’s popularity.
- **Evaluation and Insights**: Assesses model performance and provides insights into the factors influencing track popularity.

## Expected Outcomes

- **Track Popularity Prediction**: The model predicts the popularity of tracks based on audio features.
- **Feature Importance**: Identifies and ranks the audio features that most significantly influence a track’s success on Spotify.
- **Research Paper**: The findings will be documented in a research paper to contribute insights into the factors that drive track popularity in the music industry.

## Project Structure

```
data_science_project/
├── data/
│   ├── features/          # Final cleaned data used for model training and evaluation
│   └── processed/         # Processed train/test splits
├── models/
│   ├── train.py           # Model training scripts (saves trained models)
│   └── evaluate.py        # Evaluation scripts for model performance and feature importance analysis
├── notebooks/             # Jupyter Notebooks for exploratory data analysis and feature engineering
└── src/
    ├── data_preprocessing/
    │   ├── data_processing.py           # Data pipeline implementation for preprocessing
    │   └── helper_classes.py            # Custom transformers and feature engineering classes
    ├── models/
    │   ├── train.py                     # Model training and saving pipelines
    │   └── evaluate.py                  # Model evaluation and analysis
    └── split_data.py                   # Script to split raw data into training and test sets
```

## Installation

1. **Clone the Repository:**

   ```sh
   git clone https://github.com/yourusername/data_science_project.git
   cd data_science_project
   ```

2. **Create and Activate a Virtual Environment:**

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Required Packages:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Download NLTK Data (if not already available):**

   ```sh
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Usage

### Data Preprocessing and Splitting

Run the data preprocessing and splitting pipeline:

```sh
python src/split_data.py
```

This script loads raw data, applies cleaning and processing pipelines, and saves the train/test splits for further use.

### Model Training

To train the regression models and save them:

```sh
python src/models/train.py
```

This script trains multiple models (including Linear Regression, Random Forest, and XGBoost) and saves the trained models.

### Model Evaluation and Feature Importance

Evaluate the XGBoost model and display performance metrics along with the top 5 important features:

```sh
python src/test_model.py
```

The evaluation script outputs metrics such as MAE, MSE, RMSE, R², Max Error, and prints out the top 5 features based on the model’s feature importances.

## Example Evaluation Results

```
----- XGBoost Regressor Evaluation -----
MAE  (Mean Absolute Error): 16.1161
MSE  (Mean Squared Error): 399.0742
RMSE (Root Mean Squared Error): 19.9768
R²   (R-squared Score): 0.3569
Max Error: 63.7401

Top 5 Important Features:
                        feature  importance
32     track_album_release_year    0.088084
14      processed_playlist_name    0.081844
19  playlist_subgenre_last_word    0.077003
17           playlist_sentiment    0.065286
12               playlist_genre    0.065052
```

These results highlight that:
- **Track Album Release Year** is the most influential feature, implying that the year of the album release plays a major role in determining track popularity.
- **Processed Playlist Name** and attributes related to **playlist genre and sentiment** also significantly influence popularity scores.

Despite an R² of approximately 0.36, these insights serve as a strong starting point for further model improvement and deeper analysis.

## Future Work

- **Model Improvement:** Experiment with advanced hyperparameter tuning and alternative ensemble techniques to enhance model performance.
- **Feature Expansion:** Incorporate additional external data and explore richer feature interactions.
- **Error Analysis:** Investigate prediction errors to refine preprocessing and model architecture.
- **Deployment:** Package the pipeline into a deployable application for real-time track popularity analysis.

## Acknowledgements

- Thanks to Spotify for providing access to rich audio feature data through their API.
- Appreciation to the open-source community for tools and libraries like XGBoost, scikit-learn, NLTK, and more.
- Special thanks to [Your Name or Your Team's Name] for driving the project and contributing insights into the music industry's data analytics.
