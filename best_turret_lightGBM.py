import os
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from joblib import dump, load
from sklearn import set_config

# Configure scikit-learn to output pandas DataFrames
set_config(transform_output="pandas")

# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ========== IMPORTS FROM SRC ==========
from src.features import ChampionEmbeddingTransformer, TeamStrengthEncoder


class TurretPredictor:
    """
    A class to train models that predict the probability of the total number
    of turrets destroyed exceeding various thresholds.
    """
    def __init__(self, turret_thresholds=list(range(10, 15))): # e.g., Over 10, 11, 12, 13, 14
        self.models = {}
        self.team_strength_encoder = None
        self.turret_thresholds = turret_thresholds
        self.model_dir = "models/tower"

    def load_data(self, path):
        """
        Loads match data from a CSV file, processing it into a structured DataFrame.
        NOTE: Assumes the CSV has 'towers', 'result', and 'date' columns per participant.
        """
        # --- CHANGE 1: Read 'towers' as a generic 'object' first to handle text/empty cells ---
        dtype = {
            'gameid': 'string', 'side': 'string', 'champion': 'string',
            'teamname': 'string', 'towers': 'object', 'participantid': 'string',
            'result': 'int8', 'date': 'string'
        }
        try:
            df = pd.read_csv(
                path,
                usecols=dtype.keys(),
                dtype=dtype,
                parse_dates=['date'],
                na_filter=False
            )
            # --- CHANGE 2: Clean and convert the 'towers' column after loading ---
            # Coerce errors will turn any non-numeric values (like '') into NaN (Not a Number)
            # .fillna(0) then replaces these NaN values with 0
            # .astype('int16') safely converts the now-clean column to integers
            df['towers'] = pd.to_numeric(df['towers'], errors='coerce').fillna(0).astype('int16')

        except FileNotFoundError:
            print(f"Error: Data file not found at '{path}'. Cannot train models.")
            return pd.DataFrame()

        matches = []
        for game_id in df['gameid'].unique():
            game_data = df[df['gameid'] == game_id]
            if len(game_data.query("participantid not in ['100', '200']")) != 10:
                continue

            try:
                blue_team_data = game_data[game_data['side'] == 'Blue']
                red_team_data = game_data[game_data['side'] == 'Red']
                if blue_team_data.empty or red_team_data.empty: continue

                # This part now works correctly because the data is clean
                total_turrets = game_data['towers'].sum()

                matches.append({
                    'blue_picks': blue_team_data['champion'].tolist(),
                    'red_picks': red_team_data['champion'].tolist(),
                    'blue_team': blue_team_data['teamname'].iloc[0],
                    'red_team': red_team_data['teamname'].iloc[0],
                    'total_turrets': total_turrets,
                    'blue_wins': blue_team_data['result'].iloc[0],
                    'blue_side': 1,  # Blue side indicator
                    'date': blue_team_data['date'].iloc[0]
                })
            except (IndexError, Exception) as e:
                print(f"Skipping game {game_id} due to processing error: {e}")
                continue

        if not matches:
             return pd.DataFrame()
        return pd.DataFrame(matches).sort_values('date').reset_index(drop=True)

    def train_single_model(self, X, y_target, sample_weight, team_encoder, model_path):
        """Trains and saves a single LightGBM model for a specific threshold."""
        pipeline = Pipeline([
            ('features', ColumnTransformer([
                ('matchup', ChampionEmbeddingTransformer(role_aware=True), ['blue_picks', 'red_picks']),
                ('teams', team_encoder, ['blue_team', 'red_team']),
                ('side', 'passthrough', ['blue_side']),
            ], remainder='drop')),
            ('classifier', lgb.LGBMClassifier(random_state=42, verbose=-1))
        ])

        param_grid = {
            'classifier__n_estimators': [100, 250],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__num_leaves': [20, 31],
            'classifier__colsample_bytree': [0.8, 1.0],
        }

        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y_target, classifier__sample_weight=sample_weight)

        best_pipeline = grid_search.best_estimator_
        accuracy = best_pipeline.score(X, y_target, classifier__sample_weight=sample_weight)

        print(f"Model for '{y_target.name}': Accuracy = {accuracy:.2%}")
        print(f"Best Params: {grid_search.best_params_}")

        dump(best_pipeline, model_path)
        self.models[y_target.name] = best_pipeline

    def train(self, data_path='matches.csv'):
        """Main training loop to create a model for each turret threshold."""
        os.makedirs(self.model_dir, exist_ok=True)
        df = self.load_data(data_path)
        if df.empty:
            print("No data available for training.")
            return

        X = df[['blue_picks', 'red_picks', 'blue_team', 'red_team', 'blue_side']]
        y_wins = df['blue_wins']

        print("Fitting team strength encoder...")
        self.team_strength_encoder = TeamStrengthEncoder().fit(X, y_wins)

        sample_weight = np.linspace(0.2, 1.0, len(df))

        for t in self.turret_thresholds:
            label = f'over_{t}_5'
            y_target = (df['total_turrets'] > t).astype(int)
            y_target.name = label
            model_path = os.path.join(self.model_dir, f"{label}.joblib")

            print(f"\n--- Training model for {label} ---")
            self.train_single_model(X, y_target, sample_weight, self.team_strength_encoder, model_path)

# --- Example Usage ---
if __name__ == '__main__':
    # Initialize the predictor with desired turret thresholds
    predictor = TurretPredictor(turret_thresholds=list(range(10, 15)))

    # Train models
    # Your CSV must have: gameid, side, champion, teamname, turretsDestroyed, participantid, result, date
    predictor.train('matches.csv')

    print("\n✅ Training complete for all turret models.")