import pandas as pd
import numpy as np
from joblib import load
import sys
import os
from colorama import Fore, Style, init

init(autoreset=True)

# Important: Must import the custom transformers so joblib knows them
from src.features import ChampionEmbeddingTransformer, TeamStrengthEncoder, ChampionWinRateEstimator, TeamSideWinRateEstimator

def predict_ultimate(blue_picks, red_picks, blue_team, red_team):
    model_path = 'models/ultimate_winner_model.joblib'
    
    if not os.path.exists(model_path):
        print(f"{Fore.RED}Model not found at {model_path}. Please run train_ultimate.py first.")
        return

    try:
        model = load(model_path)
    except Exception as e:
        print(f"{Fore.RED}Error loading model: {e}")
        return

    # Prepare input
    data = [{
        'blue_picks': [c.strip().title() for c in blue_picks],
        'red_picks': [c.strip().title() for c in red_picks],
        'blue_team': blue_team,
        'red_team': red_team
    }]
    
    df = pd.DataFrame(data)
    
    # Predict
    try:
        probs = model.predict_proba(df)[0]
        blue_prob = probs[1]
        red_prob = probs[0]
        
        winner = "Blue" if blue_prob > 0.5 else "Red"
        confidence = max(blue_prob, red_prob)
        
        # Output
        print(f"\n{Style.BRIGHT}{Fore.CYAN}=== 🔮 ULTIMATE PREDICTION ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Matchup:{Fore.RESET} {blue_team} vs {red_team}")
        
        print(f"\n{Fore.BLUE}Blue Team Picks:{Fore.RESET} {', '.join(blue_picks)}")
        print(f"{Fore.RED}Red Team Picks:{Fore.RESET}  {', '.join(red_picks)}")
        
        print(f"\n{Style.BRIGHT}Predicted Winner: {Fore.GREEN}{winner.upper()}{Fore.RESET}")
        print(f"Confidence: {Fore.MAGENTA}{confidence:.1%}{Fore.RESET}")
        
        # Visual Bar
        bar_len = 20
        blue_chars = int(blue_prob * bar_len)
        red_chars = bar_len - blue_chars
        bar = f"{Fore.BLUE}{'█' * blue_chars}{Fore.RED}{'█' * red_chars}{Fore.RESET}"
        print(f"\n{Fore.BLUE}{blue_prob:.1%} {bar} {Fore.RED}{red_prob:.1%}")
        
    except Exception as e:
        print(f"{Fore.RED}Prediction error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage
    # You can change these values to test specific matchups
    b_picks = ["Camille", "Viego", "Galio", "Kaisa", "Nautilus"]
    r_picks = ["Renekton", "Nidalee", "Corki", "Varus", "Braum"]
    b_team = "T1"
    r_team = "Gen.G"
    
    predict_ultimate(b_picks, r_picks, b_team, r_team)
