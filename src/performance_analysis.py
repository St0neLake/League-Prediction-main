import pandas as pd
import numpy as np
from joblib import load
import os
from src.features import ChampionEmbeddingTransformer, TeamStrengthEncoder, MatchupTransformer, RoleAwareMatchupTransformer, ChampionWinRateEstimator, TeamSideWinRateEstimator

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        # Load Standard Model
        std_path = 'models/model.joblib'
        if os.path.exists(std_path):
            try:
                self.models['standard'] = load(std_path)
            except Exception as e:
                print(f"Error loading standard model: {e}")
        
        # Load Ultimate Model
        ult_path = 'models/ultimate_winner_model.joblib'
        if os.path.exists(ult_path):
            try:
                self.models['ultimate'] = load(ult_path)
            except Exception as e:
                print(f"Error loading ultimate model: {e}")

    def evaluate_on_data(self, df):
        if not self.models:
            return None

        # Prepare features - utilizing the same logic as training/inference
        if 'blue_side' not in df.columns:
            df['blue_side'] = 1

        X = df[['blue_picks', 'red_picks', 'blue_team', 'red_team', 'blue_side']]
        y_true = df['result']

        results = {
            'total_matches': len(df),
            'models': {}
        }

        for name, pipeline in self.models.items():
            try:
                stats = self._evaluate_single_model(pipeline, X, y_true)
                results['models'][name] = stats
            except Exception as e:
                print(f"Error evaluating {name} model: {e}")
                import traceback
                traceback.print_exc()

        return results

    def _evaluate_single_model(self, pipeline, X, y_true):
        # Get probabilities
        proba = pipeline.predict_proba(X)
        
        # Helper to get confidence and prediction
        blue_win_prob = proba[:, 1]
        predicted_class = (blue_win_prob > 0.5).astype(int)
        confidence = np.where(blue_win_prob > 0.5, blue_win_prob, 1 - blue_win_prob)
        
        results_df = pd.DataFrame({
            'true_class': y_true,
            'predicted_class': predicted_class,
            'confidence': confidence
        })

        # Create bins
        bins = np.arange(0.5, 1.05, 0.05)
        # Labels for bins (e.g., "50-55%", "55-60%")
        bin_labels = [f"{int(b*100)}-{int((b+0.05)*100)}%" for b in bins[:-1]]
        
        results_df['bin'] = pd.cut(results_df['confidence'], bins=bins, labels=bin_labels, include_lowest=True)

        # Group by bin and calculate metrics
        stats = []
        for bin_label in bin_labels:
            bin_data = results_df[results_df['bin'] == bin_label]
            count = len(bin_data)
            if count > 0:
                accuracy = (bin_data['true_class'] == bin_data['predicted_class']).mean()
                avg_conf = bin_data['confidence'].mean()
            else:
                accuracy = 0
                avg_conf = 0
            
            stats.append({
                'bin': bin_label,
                'accuracy': round(accuracy * 100, 1),
                'count': int(count),
                'expected_accuracy': round(avg_conf * 100, 1)
            })
        
        # Calculate overall accuracy
        overall_accuracy = (results_df['true_class'] == results_df['predicted_class']).mean()
        
        return {
            'bins': stats,
            'overall_accuracy': round(overall_accuracy * 100, 1)
        }
