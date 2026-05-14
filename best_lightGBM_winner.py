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

# ========== IMPORTS FROM SRC ==========
from src.features import ChampionEmbeddingTransformer, TeamStrengthEncoder
from src.data import load_data


class MatchPredictor:
    def __init__(self):
        self.pipeline = None
        self.matchup_transformer = None
        self.team_encoder = None

    def load_data(self, path):
        return load_data(path)


    def train(self, data_path, save_path='models/model.joblib'):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        df = self.load_data(data_path)
        if len(df) < 10:
            print(f"Need at least 10 matches, got {len(df)}")
            return

        # Add blue_side indicator (captures ~1-2% blue side advantage)
        df['blue_side'] = 1
        
        X = df[['blue_picks', 'red_picks', 'blue_team', 'red_team', 'blue_side']]
        y = df['result']
        sample_weight = np.linspace(0.1, 1, len(df))

        # --- UPDATED: Using ChampionEmbeddingTransformer (no hash collisions) ---
        pipeline = Pipeline([
            ('features', ColumnTransformer([
                ('matchup_diff', ChampionEmbeddingTransformer(role_aware=True), ['blue_picks', 'red_picks']),
                ('teams', TeamStrengthEncoder(), ['blue_team', 'red_team']),
                ('side', 'passthrough', ['blue_side']),
            ])),
            ('classifier', lgb.LGBMClassifier(random_state=42, verbose=-1))
        ])

        # --- UPDATED: Refined parameter grid ---
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__num_leaves': [31, 40],
            'classifier__max_depth': [-1, 10],
            'classifier__reg_alpha': [0, 0.1]
        }

        # --- FIX: Use TimeSeriesSplit for proper temporal cross-validation ---
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y, classifier__sample_weight=sample_weight)

        best_pipeline = grid_search.best_estimator_

        # --- FIX: Report both training accuracy and CV accuracy (properly) ---
        train_acc = best_pipeline.score(X, y)
        cv_acc = grid_search.best_score_  # This is the proper CV score

        print(f"\nTraining Accuracy: {train_acc:.2%}")
        print(f"Cross-Validation Accuracy: {cv_acc:.2%}")
        print(f"Best Parameters: {grid_search.best_params_}")

        self.pipeline = best_pipeline
        # --- Store the correct transformers ---
        self.matchup_transformer = best_pipeline.named_steps['features'].transformers_[0][1]
        self.team_encoder = best_pipeline.named_steps['features'].transformers_[1][1]
        dump(best_pipeline, save_path)

        self.print_accuracy_per_confidence(X, y)

    def load(self, model_path='models/model.joblib'):
        self.pipeline = load(model_path)
        # --- Load the correct transformers ---
        self.matchup_transformer = self.pipeline.named_steps['features'].transformers_[0][1]
        self.team_encoder = self.pipeline.named_steps['features'].transformers_[1][1]
        return self

    def predict(self, blue_picks, red_picks, blue_team, red_team):
        if len(blue_picks) != 5 or len(red_picks) != 5:
            return {'error': 'Need exactly 5 champions per team'}

        input_df = pd.DataFrame([{
            'blue_picks': [c.strip().title() for c in blue_picks],
            'red_picks': [c.strip().title() for c in red_picks],
            'blue_team': blue_team,
            'red_team': red_team,
            'blue_side': 1,  # Blue side indicator
        }])

        try:
            proba = self.pipeline.predict_proba(input_df)[0]
        except Exception as e:
            return {'error': str(e)}

        seen_champs = self.matchup_transformer.seen_champions
        known_blue = sum(1 for c in blue_picks if c.strip().title() in seen_champs)
        known_red = sum(1 for c in red_picks if c.strip().title() in seen_champs)
        quality = (known_blue + known_red) / 10

        return {
            'winner': 'Blue' if proba[1] > 0.5 else 'Red',
            'confidence': float(max(proba)),
            'quality': float(quality),
            'unknown_champions': {
                'blue': [c for c in blue_picks if c.strip().title() not in seen_champs],
                'red': [c for c in red_picks if c.strip().title() not in seen_champs]
            }
        }

    def print_accuracy_per_confidence(self, X_test, y_test):
        proba = self.pipeline.predict_proba(X_test)
        predicted_classes = np.argmax(proba, axis=1)
        confidence_scores = np.max(proba, axis=1)
        bins = np.arange(0, 1.1, 0.1)
        bin_labels = [f"{int(i*100)}-{int((i+0.1)*100)}%" for i in bins[:-1]]
        confidence_bins = pd.cut(confidence_scores, bins=bins, labels=bin_labels, include_lowest=True, right=False)
        results = pd.DataFrame({
            'confidence_bin': confidence_bins,
            'true_class': y_test,
            'predicted_class': predicted_classes
        })
        accuracy_per_bin = results.groupby('confidence_bin', observed=False).apply(
            lambda x: (x['true_class'] == x['predicted_class']).mean() if not x.empty else 0,
            include_groups=False
        ).rename('accuracy')
        count_per_bin = results.groupby('confidence_bin', observed=False).size().rename('count')
        summary = pd.concat([accuracy_per_bin, count_per_bin], axis=1).fillna(0)
        summary['accuracy'] = summary['accuracy'].astype(float)
        summary['count'] = summary['count'].astype(int)
        print("\nAccuracy per Confidence Interval:")
        print(summary)

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)

    predictor = MatchPredictor()
    predictor.train('matches.csv')
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
        print(f"\nPredicted Winner: {result['winner']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Data Quality: {result['quality']:.0%}")
        print("Unknown Champions:")
        print(f"  Blue: {result['unknown_champions']['blue']}")
        print(f"  Red: {result['unknown_champions']['red']}")