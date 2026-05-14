import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from category_encoders import TargetEncoder
from joblib import dump
import os
import warnings

# Import existing useful transformers
from src.features import ChampionEmbeddingTransformer, TeamStrengthEncoder, ChampionWinRateEstimator, TeamSideWinRateEstimator
from src.data import load_data

# Suppress warnings
warnings.filterwarnings('ignore')


def train_ultimate_model():
    print("Loading data...")
    df = load_data('matches.csv')
    
    # Needs a bit of data to work
    if len(df) < 50:
        print("Not enough data to train the ultimate model substantially.")
        return

    # Prepare features and target
    X = df[['blue_picks', 'red_picks', 'blue_team', 'red_team']]
    y = df['result']
    
    # Create the ultimate pipeline
    # 1. Champion Embeddings (Role Aware) - captures composition synergy/counter-picks
    # 2. Team Strength (General) - captures overall team skill
    # 3. Champion Win Rates - captures raw meta strength of selected champs
    # 4. Team Side Strength - captures specific side biases of teams
    
    pipeline = Pipeline([
        ('features', ColumnTransformer([
            ('embeddings', ChampionEmbeddingTransformer(role_aware=True), ['blue_picks', 'red_picks']),
            ('team_strength', TeamStrengthEncoder(), ['blue_team', 'red_team']),
            ('champ_wr', ChampionWinRateEstimator(), ['blue_picks', 'red_picks']),
            ('team_side_wr', TeamSideWinRateEstimator(), ['blue_team', 'red_team']),
        ])),
        ('classifier', lgb.LGBMClassifier(random_state=42, verbose=-1))
    ])
    
    # Parameters for Grid Search
    param_grid = {
        'classifier__n_estimators': [200, 500],
        'classifier__learning_rate': [0.01, 0.05],
        'classifier__num_leaves': [31, 63],
        'classifier__max_depth': [-1, 10],
    }
    
    print("Training Ultimate Model with TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    # We weight recent games higher
    sample_weight = np.linspace(0.5, 1.5, len(df))
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=tscv, 
        scoring='accuracy', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y, classifier__sample_weight=sample_weight)
    
    best_model = grid_search.best_estimator_
    
    print(f"\nBest Params: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.2%}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    save_path = 'models/ultimate_winner_model.joblib'
    dump(best_model, save_path)
    print(f"Model saved to {save_path}")
    
    # --- Analysis of Accuracy per Confidence ---
    print("\nCalculating Confidence Analysis...")
    input_df = X  # Use full dataset for this display, though strictly should be test set
    probs = best_model.predict_proba(input_df)
    
    # Get max probability (confidence)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    results_df = pd.DataFrame({
        'confidence': confidences,
        'correct': predictions == y
    })
    
    # Bin by confidence
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    results_df['bin'] = pd.cut(results_df['confidence'], bins=bins, labels=labels)
    
    summary = results_df.groupby('bin', observed=True)['correct'].agg(['mean', 'count'])
    summary.columns = ['Accuracy', 'Count']
    print(summary)

if __name__ == "__main__":
    train_ultimate_model()
