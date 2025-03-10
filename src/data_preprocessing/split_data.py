from typing import Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

class DataLoader:
    """Handles data loading operations"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV"""
        return pd.read_csv(self.file_path)
    
class DataSplitter:
    """Handles data splitting and saving"""
    def __init__(self, test_size: float=0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train/test sets"""
        return train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state
        )
    
    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to CSV"""
        df.to_csv(filename, index=False)

class SpotifyPipeline:
    """Orchestrates the complete data processing workflow"""
    def __init__(self, config: dict):
        self.config = config
        self.loader = DataLoader(config['file_path'])
        self.splitter = DataSplitter(
            test_size=config.get('test_size', 0.2),
            random_state=config.get('random_state', 42)
        )

    def run(self) -> None:
        """Execute complate processing pipeline"""
        # Load data
        df = self.loader.load_data()

        # Split data
        train_df, test_df = self.splitter.split(df)

        # Save results
        self.splitter.save_data(train_df, self.config['train_path'])
        self.splitter.save_data(test_df, self.config['test_path'])