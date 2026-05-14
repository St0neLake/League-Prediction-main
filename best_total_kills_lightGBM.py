#!/usr/bin/env python
"""
best_total_kills_lightGBM.py
Training script for total kills regression model
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from joblib import dump
import warnings
from sklearn import set_config

# Tell skikit-learn to output DataFrames from transformers
set_config(transform_output="pandas")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import from src
from src.data import load_data
from src.features import ChampionEmbeddingTransformer, TeamStrengthEncoder

def main():
    print("=" * 60)
    print("TRAINING TOTAL KILLS REGRESSION MODEL")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    df = load_data('matches.csv')
    print(f"Loaded {len(df)} matches")
    
    # Prepare features and target
    X = df[['blue_picks', 'red_picks', 'blue_team', 'red_team', 'blue_side']]
    y = df['total_kills']
    
    print(f"Target stats: mean={y.mean():.1f}, std={y.std():.1f}, min={y.min()}, max={y.max()}")
    
    # Create sample weights (more recent games are weighted higher)
    sample_weight = np.linspace(0.1, 1, len(df))
    
    # Build pipeline
    print("\n[2/4] Building pipeline...")
    pipeline = Pipeline([
        ('features', ColumnTransformer([
            ('matchup', ChampionEmbeddingTransformer(role_aware=True), ['blue_picks', 'red_picks']),
            ('teams', TeamStrengthEncoder(), ['blue_team', 'red_team']),
            ('side', 'passthrough', ['blue_side']),
        ], remainder='drop')),
        ('regressor', lgb.LGBMRegressor(random_state=42, verbose=-1))
    ])
    
    # Parameter grid
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__num_leaves': [31, 40, 50],
        'regressor__max_depth': [10, 15, 20]
    }
    
    # Train with TimeSeriesSplit
    print("\n[3/4] Training model with GridSearchCV...")
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=tscv, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y, regressor__sample_weight=sample_weight)
    
    # Get best model
    best_pipeline = grid_search.best_estimator_
    
    # Evaluate
    predictions = best_pipeline.predict(X)
    mae = np.mean(np.abs(predictions - y))
    rmse = np.sqrt(np.mean((predictions - y) ** 2))
    
    print(f"\n✓ Training complete!")
    print(f"  Mean Absolute Error: {mae:.2f} kills")
    print(f"  Root Mean Squared Error: {rmse:.2f} kills")
    print(f"  Best params: {grid_search.best_params_}")
    
    # Save model
    print("\n[4/4] Saving model...")
    os.makedirs('models/total_kills', exist_ok=True)
    model_path = 'models/total_kills/total_kills_model.joblib'
    dump(best_pipeline, model_path)
    print(f"✓ Saved to {model_path}")
    
    # Test prediction
    print("\n" + "=" * 60)
    print("TEST PREDICTION")
    print("=" * 60)
    test_input = pd.DataFrame([{
        'blue_picks': ["Rumble", "Nocturne", "Taliyah", "Varus", "Rakan"],
        'red_picks': ["Ornn", "Nidalee", "Tristana", "Jhin", "Bard"],
        'blue_team': "Beşiktaş Esports",
        'red_team': "BoostGate Esports",
        'blue_side': 1
    }])
    
    predicted_kills = best_pipeline.predict(test_input)[0]
    print(f"Predicted total kills: {predicted_kills:.1f}")
    print("=" * 60)

if __name__ == '__main__':
    main()
