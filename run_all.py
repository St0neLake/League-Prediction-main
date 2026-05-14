import numpy as np
import pandas as pd
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from category_encoders import TargetEncoder
from colorama import Fore, Style, init
import os

# Initialize colorama for terminal colors
init(autoreset=True)

# ========== CUSTOM TRANSFORMER CLASSES ==========
# Imported from src.features to share logic with training scripts
from src.features import TeamStrengthEncoder, ChampionHasher, MatchupTransformer, RoleAwareMatchupTransformer, ChampionEmbeddingTransformer
# Note: TeamWinrateEncoder was similar to TeamStrengthEncoder, consolidating.




class UnifiedPredictor:
    def __init__(self):
        self.models = {
            'match_winner': os.path.join('models', 'model.joblib'),
            'game_length': os.path.join('models', 'game_length', 'game_length_model.joblib'),
            'total_towers': os.path.join('models', 'tower', 'tower_model.joblib'),
            'total_kills': os.path.join('models', 'total_kills', 'total_kills_model.joblib'),
            'kill_thresholds': [os.path.join('models', 'kill_predictors', f'over_{x}_5.joblib') for x in range(20, 35)],
            'tower_thresholds': [os.path.join('models', 'tower', f'over_{x}_5.joblib') for x in [10, 11, 12, 13, 14]],
            'time_thresholds': [os.path.join('models', 'game_length', f'time_over_{x}_min.joblib') for x in range(28, 36)],
            'ultimate_winner': os.path.join('models', 'ultimate_winner_model.joblib')
        }

        # Load models with error handling
        for key in list(self.models.keys()):
            if isinstance(self.models[key], list):
                loaded = {}
                for fpath in self.models[key]:
                    try:
                        loaded[os.path.basename(fpath)] = load(fpath)  # store using filename as key
                        print(f"{Fore.GREEN}[OK] Loaded {fpath}")
                    except Exception as e:
                        print(f"{Fore.RED}[ERR] Error loading {fpath}: {str(e)}")
                self.models[key] = loaded
            else:
                try:
                    self.models[key] = load(self.models[key])
                    print(f"{Fore.GREEN}[OK] Loaded {key} model")
                except Exception as e:
                    print(f"{Fore.RED}[ERR] Error loading {key} model: {str(e)}")
                    del self.models[key]


    def predict_all(self, blue_picks, red_picks, blue_team="Unknown", red_team="Unknown"):
        input_df = pd.DataFrame([{
            'blue_picks': [c.strip().title() for c in blue_picks],
            'red_picks': [c.strip().title() for c in red_picks],
            'blue_team': blue_team,
            'red_team': red_team,
            'blue_side': 1,
            'red_side': 0
        }])

        results = {
            'teams': {
                'blue': {'name': blue_team, 'picks': blue_picks},
                'red': {'name': red_team, 'picks': red_picks}
            }
        }

        # Match winner prediction
        if 'match_winner' in self.models:
            try:
                match_proba = self.models['match_winner'].predict_proba(input_df)[0]
                results['match_winner'] = {
                    'winner': 'Blue' if match_proba[1] > 0.5 else 'Red',
                    'confidence': float(max(match_proba)),
                    'probabilities': {
                        'Blue': float(match_proba[1]),
                        'Red': float(match_proba[0])
                    }
                }
                
                # Also predict without team names (champion-only prediction)
                input_no_team = input_df.copy()
                input_no_team['blue_team'] = 'Unknown'
                input_no_team['red_team'] = 'Unknown'
                match_proba_no_team = self.models['match_winner'].predict_proba(input_no_team)[0]
                results['match_winner_no_team'] = {
                    'winner': 'Blue' if match_proba_no_team[1] > 0.5 else 'Red',
                    'confidence': float(max(match_proba_no_team)),
                    'probabilities': {
                        'Blue': float(match_proba_no_team[1]),
                        'Red': float(match_proba_no_team[0])
                    }
                }
            except Exception as e:
                results['match_winner'] = {'error': str(e)}

        # Ultimate Winner prediction
        if 'ultimate_winner' in self.models:
            try:
                # Ultimate model uses specific features but the input_df structure is compatible 
                # (UnifiedPredictor constructs input_df with same columns as train_ultimate expected)
                ult_proba = self.models['ultimate_winner'].predict_proba(input_df)[0]
                results['ultimate_winner'] = {
                    'winner': 'Blue' if ult_proba[1] > 0.5 else 'Red',
                    'confidence': float(max(ult_proba)),
                    'probabilities': {
                        'Blue': float(ult_proba[1]),
                        'Red': float(ult_proba[0])
                    }
                }
            except Exception as e:
                results['ultimate_winner'] = {'error': str(e)}

        # Game length prediction
        if 'game_length' in self.models:
            try:
                game_length = float(self.models['game_length'].predict(input_df)[0])
                results['game_length'] = {
                    'predicted_minutes': round(game_length, 1),
                    'confidence_interval': (
                        round(max(0, game_length - 2), 1),
                        round(game_length + 2, 1)
                    )
                }
            except Exception as e:
                results['game_length'] = {'error': str(e)}

        # Game length prediction
        if 'total_kills' in self.models:
            try:
                total_kills = float(self.models['total_kills'].predict(input_df)[0])
                results['total_kills'] = {
                    'predicted_kills': round(total_kills, 1),
                    'confidence_interval': (
                        round(max(0, total_kills - 2), 1),
                        round(total_kills + 2, 1)
                    )
                }
            except Exception as e:
                results['total_kills'] = {'error': str(e)}

        # Tower predictions
        if 'total_towers' in self.models:
            try:
                tower_pred = float(self.models['total_towers'].predict(input_df)[0])
                results['total_towers'] = {
                    'predicted_towers': round(tower_pred, 1),
                    'ranges': {
                        'low': max(0, round(tower_pred - 2, 1)),
                        'high': round(tower_pred + 2, 1)
                    }
                }
            except Exception as e:
                results['total_towers'] = {'error': str(e)}

        # Threshold probabilities
        results.update({
            'kill_probabilities': self._get_threshold_probs('kill_thresholds', input_df),
            'tower_probabilities': self._get_threshold_probs('tower_thresholds', input_df),
            'time_probabilities': self._get_threshold_probs('time_thresholds', input_df),
        })

        return results

    def _get_threshold_probs(self, threshold_type, input_df):
        probs = {}
        if threshold_type in self.models:
            for fname, model in self.models[threshold_type].items():
                try:
                    if threshold_type == 'time_thresholds':
                        threshold_str = fname.split('_')[2].replace('.joblib', '')
                        threshold = float(threshold_str)
                    else:
                        threshold_str = fname.split('_')[1].replace('.joblib', '').replace('_', '.')
                        threshold = float(threshold_str)

                    prob = float(model.predict_proba(input_df)[0][1])
                    probs[f'over_{threshold}'] = round(prob, 3)

                except Exception as e:
                    key = f'over_{threshold}' if 'threshold' in locals() else fname
                    probs[key] = str(e)

        return probs



# ========== VISUAL OUTPUT FORMATTING ==========
def print_predictions(result):
    """Print predictions with styled terminal output"""
    title_style = Fore.CYAN + Style.BRIGHT
    header_style = Fore.MAGENTA + Style.BRIGHT
    key_style = Fore.YELLOW
    val_style = Fore.WHITE
    pos_style = Fore.GREEN
    neg_style = Fore.RED

    print(f"\n{title_style}⚔️=== LOL MATCH PREDICTIONS ===⚔️")

    # Team information
    if 'teams' in result:
        teams = result['teams']
        print(f"\n{Fore.WHITE}{Style.BRIGHT}⚡ Teams:")
        print(f"{Fore.BLUE}Blue Team: {teams['blue']['name']}")
        print(f"  Champions: {', '.join(teams['blue']['picks'])}")
        print(f"{Fore.RED}Red Team:  {teams['red']['name']}")
        print(f"  Champions: {', '.join(teams['red']['picks'])}")

    # Match winner
    if 'match_winner' in result:
        mw = result['match_winner']
        if 'error' in mw:
            print(f"\n{neg_style}❌ Match Winner Error: {mw['error']}")
        else:
            print(f"\n{header_style}🏆 MATCH OUTCOME")
            winner_team = teams['red']['name'] if mw['winner'] == 'Red' else teams['blue']['name']
            print(f"{key_style}Predicted Winner: {Fore.GREEN}{Style.BRIGHT}{winner_team}")
            print(f"{key_style}Confidence: {pos_style if mw['confidence'] > 0.7 else neg_style}{mw['confidence']:.1%}")
            print(f"{key_style}Win Probabilities:")
            print(f"  {Fore.BLUE}{teams['blue']['name']}: {mw['probabilities']['Blue']:.1%}")
            print(f"  {Fore.RED}{teams['red']['name']}:  {mw['probabilities']['Red']:.1%}")

    # Game duration
    if 'game_length' in result:
        gl = result['game_length']
        if 'error' in gl:
            print(f"\n{neg_style}❌ Game Length Error: {gl['error']}")
        else:
            print(f"\n{header_style}⏳ GAME DURATION")
            print(f"{key_style}Predicted Length: {val_style}{gl['predicted_minutes']} minutes")
            print(f"{key_style}Confidence Range: {val_style}{gl['confidence_interval'][0]} - {gl['confidence_interval'][1]} mins")

    # Towers
    if 'total_towers' in result:
        tt = result['total_towers']
        if 'error' in tt:
            print(f"\n{neg_style}❌ Tower Prediction Error: {tt['error']}")
        else:
            print(f"\n{header_style}🏰 TOWER PREDICTIONS")
            print(f"{key_style}Expected Total: {val_style}{tt['predicted_towers']} towers")
            print(f"{key_style}Likely Range: {val_style}{tt['ranges']['low']} - {tt['ranges']['high']} towers")

    # Probability bars
    def print_bars(probs, label, emoji):
        print(f"\n{header_style}{emoji} {label.upper()} THRESHOLDS")
        for threshold, prob in sorted(probs.items(),
                                     key=lambda x: float(x[0].split('_')[1].replace('.', '_'))):
            if isinstance(prob, str):
                print(f"{neg_style}  {threshold}: {prob}")
                continue

            t_value = threshold.split('_')[1]
            bar = '▰' * int(prob * 10) + '▱' * (10 - int(prob * 10))
            color = pos_style if prob > 0.5 else Fore.YELLOW if prob > 0.3 else neg_style
            print(f"{key_style}Over {t_value} {label}: {color}{prob:.1%} {bar}")

    # Game duration
    if 'total_kills' in result:
        tks = result['total_kills']
        if 'error' in tks:
            print(f"\n{neg_style}❌ Total Kills Error: {tks['error']}")
        else:
            print(f"\n{header_style} Total Kills")
            print(f"{key_style}Predicted Total Kills: {val_style}{tks['predicted_kills']} kills")
            print(f"{key_style}Confidence Range: {val_style}{tks['confidence_interval'][0]} - {tks['confidence_interval'][1]}")

    if 'kill_probabilities' in result:
        print_bars(result['kill_probabilities'], "kills", "💀")

    if 'tower_probabilities' in result:
        print_bars(result['tower_probabilities'], "towers", "🏯")

    if 'time_probabilities' in result:
        print_bars(result['time_probabilities'], "time", "⏱️") # Changed emoji

    print(f"\n{title_style}⚔️=== END OF PREDICTIONS ===⚔️")

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    predictor = UnifiedPredictor()

    result = predictor.predict_all(
        blue_picks=["Rumble", "Nocturne", "Taliyah", "Varus", "Rakan"],
        red_picks=["Ornn", "Nidalee", "Tristana", "Jhin", "Bard"],
        blue_team="Beşiktaş Esports",
        red_team="BoostGate Esports"
    )

    print_predictions(result)