import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from category_encoders import TargetEncoder  # pip install category_encoders
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from joblib import dump, load
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ChampionHasher(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=20):
        self.n_features = n_features
        self.hasher = FeatureHasher(n_features=n_features, input_type='string')
        self.seen_champions = set()

    def fit(self, X, y=None):
        for _, row in X.iterrows():
            self._process_team(row['blue_picks'])
            self._process_team(row['red_picks'])
        return self

    def _process_team(self, picks):
        for champ in picks:
            normalized = champ.strip().title()
            self.seen_champions.add(normalized)

    def transform(self, X):
        def hash_team(picks):
            normalized = [c.strip().title() for c in picks]
            return self.hasher.transform([normalized]).toarray()[0]
        blue_features = X['blue_picks'].apply(hash_team)
        red_features = X['red_picks'].apply(hash_team)
        return np.hstack([np.vstack(blue_features), np.vstack(red_features)])


class TeamEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = TargetEncoder(smoothing=10)

    def fit(self, X, y):
        teams = pd.concat([X['blue_team'], X['red_team']])
        y_series = pd.Series(y)
        results = pd.concat([y_series, pd.Series(1 - y_series)])
        self.encoder.fit(teams.values.reshape(-1, 1), results)
        return self

    def transform(self, X):
        blue_enc = self.encoder.transform(X['blue_team'].values.reshape(-1, 1))
        red_enc = self.encoder.transform(X['red_team'].values.reshape(-1, 1))
        return np.hstack([blue_enc, red_enc])


class EnsembleMatchPredictor:
    def __init__(self):
        self.pipeline = None
        self.champion_hasher = None
        self.team_encoder = None

    def load_data(self, path):
        dtype = {
            'gameid': 'string',
            'side': 'string',
            'champion': 'string',
            'teamname': 'string',
            'result': 'int8',
            'date': 'string',
            'participantid': 'string'
        }
        df = pd.read_csv(
            path,
            usecols=['gameid', 'side', 'champion', 'teamname', 'result', 'date', 'participantid'],
            dtype=dtype,
            parse_dates=['date'],
            na_filter=False
        )
        matches = []
        for game_id in df['gameid'].unique():
            game_data = df[df['gameid'] == game_id]
            game_data = game_data[~game_data['participantid'].isin(['100', '200'])]
            if len(game_data) != 10:
                continue
            try:
                blue_team = game_data[game_data['side'] == 'Blue']
                red_team = game_data[game_data['side'] == 'Red']
                blue_picks = blue_team['champion'].tolist()
                red_picks = red_team['champion'].tolist()
                if len(blue_picks) != 5 or len(red_picks) != 5:
                    continue
                matches.append({
                    'blue_picks': blue_picks,
                    'red_picks': red_picks,
                    'blue_team': blue_team['teamname'].iloc[0],
                    'red_team': red_team['teamname'].iloc[0],
                    'blue_side': 1,
                    'red_side': 0,
                    'result': blue_team['result'].iloc[0],
                    'date': blue_team['date'].iloc[0]
                })
            except Exception:
                continue
        return pd.DataFrame(matches)

    def train(self, data_path, save_path='best_ensemble_model.joblib'):
        df = self.load_data(data_path)
        if len(df) < 10:
            print(f"Need at least 10 matches, got {len(df)}")
            return

        X = df[['blue_picks', 'red_picks', 'blue_team', 'red_team', 'blue_side', 'red_side']]
        y = df['result']

        features_transformer = ColumnTransformer([
            ('champions', ChampionHasher(), ['blue_picks', 'red_picks']),
            ('teams', TeamEncoder(), ['blue_team', 'red_team']),
            ('sides', 'passthrough', ['blue_side', 'red_side'])
        ])

        # Base models
        hgb = HistGradientBoostingClassifier(
            max_iter=500,
            early_stopping=True,
            random_state=42,
            min_samples_leaf=20,
            l2_regularization=0.5
        )
        svm = SVC(probability=True, random_state=42)
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')

        # Voting Ensemble with 4 models
        voting_ensemble = VotingClassifier(
            estimators=[('hgb', hgb), ('svm', svm), ('rf', rf), ('xgb', xgb)],
            voting='soft'
        )
        voting_pipeline = Pipeline([
            ('features', features_transformer),
            ('classifier', voting_ensemble)
        ])

        # Expanded grid for the Voting ensemble
        voting_param_grid = {
            'classifier__hgb__learning_rate': [0.1, 0.15],
            'classifier__hgb__max_depth': [3, None],
            'classifier__svm__C': [1, 10],
            'classifier__svm__gamma': ['scale'],  # using default 'scale'
            'classifier__rf__n_estimators': [200, 300],
            'classifier__rf__max_depth': [10, 20],
            'classifier__xgb__learning_rate': [0.1, 0.15],
            'classifier__xgb__max_depth': [3, 5]
        }

        # Nested CV: inner CV with 2 splits, outer CV with 3 splits
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid_search = GridSearchCV(voting_pipeline, voting_param_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1)
        nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring="accuracy", n_jobs=-1)
        print(f"\nNested CV Accuracy: {nested_scores.mean():.2%}")

        # Fit grid search on the entire dataset
        grid_search.fit(X, y)
        best_voting_score = grid_search.best_score_
        best_voting_params = grid_search.best_params_
        print("\nVoting Ensemble Best Parameters (Refined):")
        print(best_voting_params)
        print(f"Voting Ensemble Best Cross-Validated Accuracy (Inner CV): {best_voting_score:.2%}")

        # Generate cross-validated predictions (using outer CV) for further evaluation
        y_pred_proba = cross_val_predict(grid_search.best_estimator_, X, y, cv=outer_cv, method='predict_proba')
        y_pred = np.argmax(y_pred_proba, axis=1)
        cv_accuracy = (y_pred == np.array(y)).mean()
        print(f"\nCross-Validated Training Accuracy (Best Estimator): {cv_accuracy:.2%}")

        cm = confusion_matrix(y, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y, y_pred))

        grid_search.best_estimator_.fit(X, y)
        self.pipeline = grid_search.best_estimator_
        self.champion_hasher = self.pipeline.named_steps['features'].transformers_[0][1]
        self.team_encoder = self.pipeline.named_steps['features'].transformers_[1][1]
        dump(self.pipeline, save_path)
        self.print_accuracy_per_confidence(y_pred_proba, y)

    def load(self, model_path='best_ensemble_model.joblib'):
        self.pipeline = load(model_path)
        self.champion_hasher = self.pipeline.named_steps['features'].transformers_[0][1]
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
            'blue_side': 1,
            'red_side': 0
        }])
        try:
            proba = self.pipeline.predict_proba(input_df)[0]
        except Exception as e:
            return {'error': str(e)}
        known_blue = sum(1 for c in blue_picks if c.strip().title() in self.champion_hasher.seen_champions)
        known_red = sum(1 for c in red_picks if c.strip().title() in self.champion_hasher.seen_champions)
        quality = (known_blue + known_red) / 10
        return {
            'winner': 'Blue' if proba[1] > 0.5 else 'Red',
            'confidence': float(max(proba)),
            'quality': float(quality),
            'unknown_champions': {
                'blue': [c for c in blue_picks if c.strip().title() not in self.champion_hasher.seen_champions],
                'red': [c for c in red_picks if c.strip().title() not in self.champion_hasher.seen_champions]
            }
        }

    def print_accuracy_per_confidence(self, proba, true_y):
        predicted_classes = np.argmax(proba, axis=1)
        confidence_scores = np.max(proba, axis=1)
        bins = np.arange(0, 1.1, 0.1)
        bin_labels = [f"{i*10}-{(i+1)*10}%" for i in range(len(bins)-1)]
        confidence_bins = pd.cut(confidence_scores, bins=bins, labels=bin_labels, include_lowest=True)
        results = pd.DataFrame({
            'confidence_bin': confidence_bins,
            'true_class': true_y,
            'predicted_class': predicted_classes
        })
        accuracy_per_bin = results.groupby('confidence_bin').agg(
            accuracy=('true_class', lambda x: (x == results.loc[x.index, 'predicted_class']).mean()),
            count=('true_class', 'count')
        )
        accuracy_per_bin = accuracy_per_bin.reindex(bin_labels, fill_value=0)
        print("\nAccuracy per Confidence Interval:")
        print(accuracy_per_bin)


if __name__ == "__main__":
    predictor = EnsembleMatchPredictor()
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
        print(f"Predicted Winner: {result['winner']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Data Quality: {result['quality']:.0%}")
        print("Unknown Champions:")
        print(f"Blue: {result['unknown_champions']['blue']}")
        print(f"Red: {result['unknown_champions']['red']}")
