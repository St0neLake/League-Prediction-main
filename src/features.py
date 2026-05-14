import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from category_encoders import TargetEncoder
from src.roles import RoleSorter


class ChampionEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms champion picks into one-hot encoded vectors with role-awareness.
    Unlike FeatureHasher, this builds a vocabulary during fit() and uses
    exact one-hot encoding, avoiding hash collisions.
    """
    def __init__(self, role_aware=True):
        self.role_aware = role_aware
        self.champion_to_idx = {}  # Vocabulary: champion -> index
        self.n_features = 0
        self.roles = ['top', 'jng', 'mid', 'bot', 'sup']
        self.sorter = RoleSorter() if role_aware else None
        self.seen_champions = set()

    def fit(self, X, y=None):
        """Build vocabulary from all champions in training data."""
        all_champions = set()
        
        for _, row in X.iterrows():
            for team in ['blue_picks', 'red_picks']:
                if team in row:
                    for champ in row[team]:
                        normalized = champ.strip().title()
                        self.seen_champions.add(normalized)
                        if self.role_aware:
                            for role in self.roles:
                                all_champions.add(f"{role}_{normalized}")
                        else:
                            all_champions.add(normalized)
        
        # Create vocabulary mapping
        for idx, champ in enumerate(sorted(all_champions)):
            self.champion_to_idx[champ] = idx
        
        self.n_features = len(self.champion_to_idx)
        return self

    def _encode_team(self, picks, is_blue=True):
        """Encode a team's picks into a one-hot vector."""
        vec = np.zeros(self.n_features)
        
        if self.role_aware and self.sorter:
            sorted_picks = self.sorter.sort(picks)
            for role, champ in zip(self.roles, sorted_picks):
                key = f"{role}_{champ.strip().title()}"
                if key in self.champion_to_idx:
                    vec[self.champion_to_idx[key]] = 1.0 if is_blue else -1.0
        else:
            for champ in picks:
                key = champ.strip().title()
                if key in self.champion_to_idx:
                    vec[self.champion_to_idx[key]] = 1.0 if is_blue else -1.0
        
        return vec

    def transform(self, X):
        """Transform picks into matchup difference vectors (blue - red)."""
        def encode_row(row):
            blue_vec = self._encode_team(row['blue_picks'], is_blue=True)
            red_vec = self._encode_team(row['red_picks'], is_blue=False)
            # Return blue - red difference (red already negated in _encode_team)
            return blue_vec + red_vec

        result = X.apply(encode_row, axis=1)
        return np.vstack(result)


class TeamStrengthEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes team names based on their historical win rate. This encoder is
    fitted once on the overall match results (win/loss).
    """
    def __init__(self, smooth=15):
        self.smooth = smooth
        self.encoder = TargetEncoder(smoothing=smooth)

    def fit(self, X, y_wins):
        all_teams = pd.concat([X['blue_team'], X['red_team']], ignore_index=True)
        # Convert to numpy array of objects/strings to ensure compatibility
        all_teams = all_teams.astype(object).values.reshape(-1, 1)
        
        # Blue wins are y_wins, red wins are the inverse (1 - y_wins)
        all_wins = pd.concat([y_wins, 1 - y_wins], ignore_index=True).values
        self.encoder.fit(all_teams, all_wins)
        return self

    def transform(self, X):
        # Transform blue and red team names into their encoded win rates
        blue_enc = self.encoder.transform(X['blue_team'].astype(object).values.reshape(-1, 1))
        red_enc = self.encoder.transform(X['red_team'].astype(object).values.reshape(-1, 1))
        return np.hstack([blue_enc, red_enc])

class ChampionHasher(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=20):
        self.n_features = n_features
        self.hasher = FeatureHasher(n_features=n_features, input_type='string')
        self.seen_champions = set()

    def fit(self, X, y=None):
        for _, row in X.iterrows():
            for team in ['blue_picks', 'red_picks']:
                if team in row:
                    for champ in row[team]:
                        self.seen_champions.add(champ.strip().title())
        return self

    def transform(self, X):
        def hash_team(picks):
            return self.hasher.transform([[c.strip().title() for c in picks]]).toarray()[0]
        blue = np.vstack(X['blue_picks'].apply(hash_team))
        red = np.vstack(X['red_picks'].apply(hash_team))
        return np.hstack([blue, red])

class MatchupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=300):
        self.n_features = n_features
        self.hasher = FeatureHasher(n_features=self.n_features, input_type='string', alternate_sign=True)
        self.seen_champions = set()

    def fit(self, X, y=None):
        for _, row in X.iterrows():
            for champ in row['blue_picks']:
                self.seen_champions.add(champ.strip().title())
            for champ in row['red_picks']:
                self.seen_champions.add(champ.strip().title())
        return self

    def transform(self, X):
        def hash_team(picks):
            normalized = [c.strip().title() for c in picks]
            return self.hasher.transform([normalized]).toarray()[0]

        blue_hashes = np.vstack(X['blue_picks'].apply(hash_team))
        red_hashes = np.vstack(X['red_picks'].apply(hash_team))

        return blue_hashes - red_hashes

class RoleAwareMatchupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=300):
        self.n_features = n_features
        self.hasher = FeatureHasher(n_features=self.n_features, input_type='string', alternate_sign=True)
        self.sorter = RoleSorter()  # Initialize RoleSorter
        self.seen_champions = set()

    def fit(self, X, y=None):
        for _, row in X.iterrows():
            for team in ['blue_picks', 'red_picks']:
                if team in row:
                    for champ in row[team]:
                        self.seen_champions.add(champ.strip().title())
        return self

    def transform(self, X):
        def sort_and_hash_diff(row):
            # Sort champions by role
            blue_sorted = self.sorter.sort(row['blue_picks'])
            red_sorted = self.sorter.sort(row['red_picks'])
            
            # Hash aligned pairs (Top vs Top, etc.)
            # Instead of one big bag diff, we construct a feature vector that strictly respects roles.
            # However, FeatureHasher is bag-of-words. 
            # Strategy: Prefix role to champion name "TOP_Gnar", "MID_Ahri" before hashing.
            
            blue_tagged = [f"{r}_{c.strip().title()}" for r, c in zip(self.sorter.roles, blue_sorted)]
            red_tagged = [f"{r}_{c.strip().title()}" for r, c in zip(self.sorter.roles, red_sorted)]
            
            blue_vec = self.hasher.transform([blue_tagged]).toarray()[0]
            red_vec = self.hasher.transform([red_tagged]).toarray()[0]
            
            return blue_vec - red_vec

        # Apply to all rows
        diffs = X.apply(sort_and_hash_diff, axis=1)
        return np.vstack(diffs)


class ChampionWinRateEstimator(BaseEstimator, TransformerMixin):
    """
    Calculates the aggregate win rate of the team's composition based on
    historical champion win rates.
    """
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.champion_stats = {}
        self.global_mean = 0.5

    def fit(self, X, y=None):
        # Flatten all picks and outcomes
        # X is expected to have 'blue_picks' and 'red_picks'
        # y is expected to be the target (1 for Blue win, 0 for Red win)
        
        # Count wins and total games for each champion
        champ_wins = {}
        champ_games = {}
        
        if y is None:
            return self

        # Iterate X and y simultaneously to avoid index mismatch
        for (_, row), result in zip(X.iterrows(), y):
            # Blue picks
            for champ in row['blue_picks']:
                c = champ.strip().title()
                champ_games[c] = champ_games.get(c, 0) + 1
                if result == 1:
                    champ_wins[c] = champ_wins.get(c, 0) + 1
            
            # Red picks
            for champ in row['red_picks']:
                c = champ.strip().title()
                champ_games[c] = champ_games.get(c, 0) + 1
                if result == 0: # Red won
                     champ_wins[c] = champ_wins.get(c, 0) + 1

        # Calculate smoothed win rates
        self.global_mean = sum(champ_wins.values()) / sum(champ_games.values()) if sum(champ_games.values()) > 0 else 0.5
        
        for champ, games in champ_games.items():
            wins = champ_wins.get(champ, 0)
            # Bayseian smoothing
            smoothed_wr = (wins + self.smoothing * self.global_mean) / (games + self.smoothing)
            self.champion_stats[champ] = smoothed_wr
            
        return self

    def transform(self, X):
        # Calculate the average historical winrate of the composition for blue and red
        
        blue_wr_scores = []
        red_wr_scores = []
        
        for _, row in X.iterrows():
            # Blue Team Score
            b_score = 0
            for champ in row['blue_picks']:
                c = champ.strip().title()
                b_score += self.champion_stats.get(c, self.global_mean)
            blue_wr_scores.append(b_score / 5) # Average WR of the 5 champs
            
            # Red Team Score
            r_score = 0
            for champ in row['red_picks']:
                c = champ.strip().title()
                r_score += self.champion_stats.get(c, self.global_mean)
            red_wr_scores.append(r_score / 5)
            
        return np.column_stack((blue_wr_scores, red_wr_scores))


class TeamSideWinRateEstimator(BaseEstimator, TransformerMixin):
    """
    Calculates win rate for teams specifically on Blue vs Red side.
    """
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.blue_side_stats = {} # team -> winrate on blue
        self.red_side_stats = {}  # team -> winrate on red
        self.global_mean = 0.5

    def fit(self, X, y=None):
        # X has 'blue_team', 'red_team'
        # y is 1 for Blue win, 0 for Red win
        
        blue_wins = {}
        blue_games = {}
        red_wins = {}
        red_games = {}
        
        for (_, row), res in zip(X.iterrows(), y):
            b_team = row['blue_team']
            r_team = row['red_team']
            
            # Blue side stats
            blue_games[b_team] = blue_games.get(b_team, 0) + 1
            if res == 1:
                blue_wins[b_team] = blue_wins.get(b_team, 0) + 1
            
            # Red side stats
            red_games[r_team] = red_games.get(r_team, 0) + 1
            if res == 0:
                red_wins[r_team] = red_wins.get(r_team, 0) + 1
                
        self.global_mean = y.mean()
        
        # Calculate smoothed stats
        for team, games in blue_games.items():
            wins = blue_wins.get(team, 0)
            self.blue_side_stats[team] = (wins + self.smoothing * self.global_mean) / (games + self.smoothing)
            
        for team, games in red_games.items():
            wins = red_wins.get(team, 0)
            self.red_side_stats[team] = (wins + self.smoothing * (1 - self.global_mean)) / (games + self.smoothing)
            
        return self

    def transform(self, X):
        blue_side_probs = []
        red_side_probs = []
        
        for _, row in X.iterrows():
            b_team = row['blue_team']
            r_team = row['red_team']
            
            blue_side_probs.append(self.blue_side_stats.get(b_team, self.global_mean))
            red_side_probs.append(self.red_side_stats.get(r_team, 1 - self.global_mean))
            
        return np.column_stack((blue_side_probs, red_side_probs))
