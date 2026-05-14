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


class GameLengthPredictor:
    """
    A class to train models that predict the probability of game length
    exceeding various time thresholds.
    """
    def __init__(self, time_thresholds=list(range(28, 36))):
        self.models = {}
        self.team_strength_encoder = None
        self.time_thresholds = time_thresholds
        self.model_dir = "models/game_length"

    def load_data(self, path):
        """
        Loads match data from a CSV file, processing it into a structured DataFrame.
        NOTE: Assumes the CSV contains 'gamelength', 'result' (1 for blue win), and 'date'.
        """
        dtype = {
            'gameid': 'string', 'side': 'string', 'champion': 'string',
            'teamname': 'string', 'gamelength': 'int32', 'participantid': 'string',
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
        except FileNotFoundError:
            print(f"Error: Data file not found at '{path}'. Cannot train models.")
            return pd.DataFrame()

        matches = []
        for game_id in df['gameid'].unique():
            game_data = df[df['gameid'] == game_id]
            # Ensure complete data for a standard 5v5 match
            if len(game_data.query("participantid not in ['100', '200']")) != 10:
                continue

            try:
                blue_team_data = game_data[game_data['side'] == 'Blue']
                red_team_data = game_data[game_data['side'] == 'Red']
                if blue_team_data.empty or red_team_data.empty: continue

                matches.append({
                    'blue_picks': blue_team_data['champion'].tolist(),
                    'red_picks': red_team_data['champion'].tolist(),
                    'blue_team': blue_team_data['teamname'].iloc[0],
                    'red_team': red_team_data['teamname'].iloc[0],
                    'gamelength_min': blue_team_data['gamelength'].iloc[0] / 60,
                    'blue_wins': blue_team_data['result'].iloc[0],
                    'blue_side': 1,  # Blue side indicator
                    'date': blue_team_data['date'].iloc[0]
                })
            except (IndexError, Exception) as e:
                print(f"Skipping game {game_id} due to processing error: {e}")
                continue

        if not matches:
             return pd.DataFrame()
        # Sort by date to apply time-based sample weighting later
        return pd.DataFrame(matches).sort_values('date').reset_index(drop=True)

    def train_single_model(self, X, y_target, sample_weight, team_encoder, model_path):
        """Trains and saves a single LightGBM model for a specific time threshold."""
        pipeline = Pipeline([
            ('features', ColumnTransformer([
                ('matchup', ChampionEmbeddingTransformer(role_aware=True), ['blue_picks', 'red_picks']),
                ('teams', team_encoder, ['blue_team', 'red_team']),
                ('side', 'passthrough', ['blue_side']),
            ], remainder='drop')),
            ('classifier', lgb.LGBMClassifier(random_state=42, verbose=-1))
        ])

        # Hyperparameter grid for GridSearchCV
        param_grid = {
            'classifier__n_estimators': [100, 250],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__num_leaves': [20, 31],
            'classifier__colsample_bytree': [0.8, 1.0],
        }

        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
        # Pass sample weights directly to the classifier step
        grid_search.fit(X, y_target, classifier__sample_weight=sample_weight)

        best_pipeline = grid_search.best_estimator_
        accuracy = best_pipeline.score(X, y_target, classifier__sample_weight=sample_weight)

        print(f"Model for '{y_target.name}': Accuracy = {accuracy:.2%}")
        print(f"Best Params: {grid_search.best_params_}")

        dump(best_pipeline, model_path)
        self.models[y_target.name] = best_pipeline

    def train(self, data_path='matches.csv'):
        """Main training loop to create a model for each time threshold."""
        os.makedirs(self.model_dir, exist_ok=True)
        df = self.load_data(data_path)
        if df.empty:
            print("No data available for training.")
            return

        X = df[['blue_picks', 'red_picks', 'blue_team', 'red_team', 'blue_side']]
        y_wins = df['blue_wins']

        # 1. Fit the TeamStrengthEncoder ONCE on win/loss data
        print("Fitting team strength encoder...")
        self.team_strength_encoder = TeamStrengthEncoder().fit(X, y_wins)

        # 2. Create sample weights to give more importance to recent games
        sample_weight = np.linspace(0.2, 1.0, len(df))

        # 3. Train a separate model for each time threshold
        for t in self.time_thresholds:
            label = f'over_{t}_min'
            y_target = (df['gamelength_min'] > t).astype(int)
            y_target.name = label
            model_path = os.path.join(self.model_dir, f"time_{label}.joblib")
            print(f"\n--- Training model for {label} ---")
            # Pass the single, pre-fitted team encoder to each training run
            self.train_single_model(X, y_target, sample_weight, self.team_strength_encoder, model_path)

    def load(self):
        """Loads all trained model files from the model directory."""
        print(f"\nLoading models from '{self.model_dir}'...")
        for t in self.time_thresholds:
            label = f'over_{t}_min'
            model_file = os.path.join(self.model_dir, f"time_{label}.joblib")
            if os.path.exists(model_file):
                self.models[label] = load(model_file)
            else:
                print(f"Warning: Model file not found for '{label}'")
        return self

    def predict(self, blue_picks, red_picks, blue_team, red_team):
        """
        Predicts game length probabilities for a given matchup.
        Returns probabilities, data quality, and any champions not seen in training.
        """
        if not self.models:
            return {'error': 'No models are loaded. Please run train() or load() first.'}
        if len(blue_picks) != 5 or len(red_picks) != 5:
            return {'error': 'Each team must have exactly 5 champion picks.'}

        input_df = pd.DataFrame([{'blue_picks': blue_picks, 'red_picks': red_picks,
                                'blue_team': blue_team, 'red_team': red_team, 'blue_side': 1}])
        probabilities = {}
        for label, model in self.models.items():
            try:
                # Predict probability of the positive class (game time > threshold)
                proba = model.predict_proba(input_df)[0][1]
                probabilities[label] = round(proba, 3)
            except Exception as e:
                return {'error': f"Prediction failed for {label}: {e}"}

        # Get champion vocabulary from the first loaded model to assess data quality
        model_for_info = next(iter(self.models.values()))
        seen_champs = model_for_info.named_steps['features'].named_transformers_['matchup'].seen_champions
        known_blue = sum(1 for c in blue_picks if c.strip().title() in seen_champs)
        known_red = sum(1 for c in red_picks if c.strip().title() in seen_champs)

        return {
            'probabilities': probabilities,
            'data_quality': (known_blue + known_red) / 10.0,
            'unknown_champions': {
                'blue': [c for c in blue_picks if c.strip().title() not in seen_champs],
                'red': [c for c in red_picks if c.strip().title() not in seen_champs]
            }
        }

# --- Example Usage ---
if __name__ == '__main__':
    # Initialize the predictor
    predictor = GameLengthPredictor(time_thresholds=list(range(28, 36)))

    # Train models (this will take time and requires a 'matches.csv' file)
    # The CSV must have: gameid, side, champion, teamname, gamelength, participantid, result, date
    predictor.train('matches.csv')

    # Load the trained models from disk
    predictor.load()

    # Example prediction
    result = predictor.predict(
        blue_picks=["Darius", "Nidalee", "Jayce", "Ezreal", "Alistar"],
        red_picks=["Ambessa", "Skarner", "Aurora", "Varus", "Rakan"],
        blue_team="BoostGate Esports",
        red_team="BBL Dark Passage"
    )

    # Print the results
    if 'error' in result:
        print(f"\nPrediction Error: {result['error']}")
    else:
        print("\n--- Prediction Results ---")
        print("Probabilities for Game Length Over Thresholds:")
        # Sort probabilities by time for clear presentation
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: int(x[0].split('_')[1]))
        for k, v in sorted_probs:
            print(f"  - {k.replace('_', ' ').replace(' min', ' minutes')}: {v:.1%}")

        print(f"\nData Quality (known champions): {result['data_quality']:.0%}")
        if result['unknown_champions']['blue'] or result['unknown_champions']['red']:
            print("Unknown Champions Encountered:")
            if result['unknown_champions']['blue']:
                print(f"  - Blue: {', '.join(result['unknown_champions']['blue'])}")
            if result['unknown_champions']['red']:
                print(f"  - Red: {', '.join(result['unknown_champions']['red'])}")