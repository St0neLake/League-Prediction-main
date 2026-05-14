import os
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
import warnings
from sklearn import set_config

# Tell scikit-learn to output DataFrames from transformers
set_config(transform_output="pandas")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ========== IMPORTS FROM SRC ==========
from src.features import ChampionEmbeddingTransformer, TeamStrengthEncoder



class KillsPredictor:
    def __init__(self):
        self.models = {}
        # Pre-fitted encoder will be stored here
        self.team_winrate_encoder = None
        self.thresholds = [x + 0.5 for x in range(20, 35)]

    # --- UPDATED: Assumes your CSV has 'kills' and 'result' (for win/loss) columns ---
    def load_data(self, path):
        # NOTE: Add 'kills' to your dtype and usecols if it's not already there.
        dtype = {
            'gameid': 'string', 'side': 'string', 'champion': 'string', 'teamname': 'string',
            'result': 'int8', # Assumed to be 1 for win, 0 for loss
            'kills': 'int8', # Assumed to be the kills per player
            'date': 'string', 'participantid': 'string'
        }
        df = pd.read_csv(path, usecols=dtype.keys(), dtype=dtype, parse_dates=['date'], na_filter=False)
        matches = []
        for game_id in df['gameid'].unique():
            game_data = df[df['gameid'] == game_id]
            game_data = game_data[~game_data['participantid'].isin(['100', '200'])]
            if len(game_data) != 10: continue
            try:
                blue_team = game_data[game_data['side'] == 'Blue']
                red_team = game_data[game_data['side'] == 'Red']
                if blue_team.empty or red_team.empty: continue

                matches.append({
                    'blue_picks': blue_team['champion'].tolist(),
                    'red_picks': red_team['champion'].tolist(),
                    'blue_team': blue_team['teamname'].iloc[0],
                    'red_team': red_team['teamname'].iloc[0],
                    'blue_wins': blue_team['result'].iloc[0], # Blue's result is the match outcome
                    'total_kills': game_data['kills'].sum(), # Sum of all player kills
                    'blue_side': 1,  # Blue side indicator
                    'date': blue_team['date'].iloc[0]
                })
            except Exception as e:
                print(f"Error processing game {game_id}: {e}")
                continue
        return pd.DataFrame(matches).sort_values('date')

    # --- UPDATED: Accepts a pre-fitted encoder ---
    def train_single_lgbm(self, X, y_target, sample_weight, team_encoder_instance, model_path):
        pipeline = Pipeline([
            ('features', ColumnTransformer([
                ('matchup', ChampionEmbeddingTransformer(role_aware=True), ['blue_picks', 'red_picks']),
                ('teams', team_encoder_instance, ['blue_team', 'red_team']),
                ('side', 'passthrough', ['blue_side']),
            ], remainder='drop')),
            ('classifier', lgb.LGBMClassifier(random_state=42, verbose=-1))
        ])

        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__num_leaves': [31, 40]
        }

        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y_target, classifier__sample_weight=sample_weight)

        best_pipeline = grid_search.best_estimator_
        train_acc = best_pipeline.score(X, y_target)

        print(f"\nAccuracy for {y_target.name}: {train_acc:.2%}")
        print(f"Best Parameters: {y_target.name,}: {grid_search.best_params_}")

        dump(best_pipeline, model_path)
        self.models[y_target.name] = best_pipeline

    def train(self, data_path, save_path='models/kill_predictors'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        df = self.load_data(data_path)
        if len(df) < 10: return

        X = df[['blue_picks', 'red_picks', 'blue_team', 'red_team', 'blue_side']]
        y_wins = df['blue_wins']

        # --- FIX: Fit the TeamStrengthEncoder ONCE on the win/loss data ---
        self.team_winrate_encoder = TeamStrengthEncoder().fit(X, y_wins)

        sample_weight = np.linspace(0.1, 1, len(df))

        for t in self.thresholds:
            label = f'over_{str(t).replace(".", "_")}'
            y_target = (df['total_kills'] > t).astype(int)
            y_target.name = label
            model_path = os.path.join(save_path, f"{label}.joblib")

            # Pass the pre-fitted encoder to the training function
            self.train_single_lgbm(X, y_target, sample_weight, self.team_winrate_encoder, model_path)

    def load(self, model_path='models'):
        # The main models are loaded here. The single team encoder is part of each model.
        for t in self.thresholds:
            label = f'over_{str(t).replace(".", "_")}'
            model_file = os.path.join(model_path, f"{label}.joblib")
            if os.path.exists(model_file):
                self.models[label] = load(model_file)
        return self

    def predict(self, blue_picks, red_picks, blue_team, red_team):
        if len(blue_picks) != 5: return {'error': 'Need exactly 5 champions per team'}
        input_df = pd.DataFrame([{'blue_picks': blue_picks, 'red_picks': red_picks, 'blue_team': blue_team, 'red_team': red_team, 'blue_side': 1}])

        results = {}
        model_for_info = next(iter(self.models.values()), None)
        if not model_for_info: return {'error': 'No models are loaded.'}

        for label, model in self.models.items():
            try:
                proba = model.predict_proba(input_df)[0][1]
                results[label] = round(proba, 3)
            except Exception as e: return {'error': str(e)}

        seen_champs = model_for_info.named_steps['features'].named_transformers_['matchup'].seen_champions
        known_blue = sum(1 for c in blue_picks if c.strip().title() in seen_champs)
        known_red = sum(1 for c in red_picks if c.strip().title() in seen_champs)
        quality = (known_blue + known_red) / 10

        return {'probabilities': results, 'quality': quality, 'unknown_champions': {
            'blue': [c for c in blue_picks if c.strip().title() not in seen_champs],
            'red': [c for c in red_picks if c.strip().title() not in seen_champs]
        }}

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    predictor = KillsPredictor()
    predictor.train('matches.csv') # Assumes 'matches.csv' contains 'kills' and 'result' columns
    predictor.load()

    result = predictor.predict(
        blue_picks=["Kennen", "Sejuani", "Yone", "Draven", "Nautilus"],
        red_picks=["Renekton", "Lillia", "Zed", "Jhin", "Leona"],
        blue_team="FunPlus Phoenix",
        red_team="Ultra Prime"
    )

    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print("\nProbabilities for Over Kill Totals:")
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: float(x[0].split("_")[1].replace("_", ".")))
        for k, v in sorted_probs:
            print(f"{k.replace('_', '.')} kills: {v:.1%}")

        print(f"\nData Quality: {result['quality']:.0%}")
        if result['unknown_champions']['blue'] or result['unknown_champions']['red']:
            print("Unknown Champions:")
            print(f"  Blue: {result['unknown_champions']['blue']}")
            print(f"  Red: {result['unknown_champions']['red']}")