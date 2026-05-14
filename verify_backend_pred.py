import pandas as pd
from run_all import UnifiedPredictor

def check_backend():
    print("Initializing UnifiedPredictor...")
    try:
        predictor = UnifiedPredictor()
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return

    # specific test for Kai'Sa
    print("\nTesting prediction with Kai'Sa...")
    try:
        blue_picks = ["Kai'Sa", "Garen", "Lux", "Ahri", "Viego"]
        red_picks = ["Ashe", "Darius", "Zed", "Lulu", "Lee Sin"]
        
        result = predictor.predict_all(
            blue_picks=blue_picks,
            red_picks=red_picks,
            blue_team="T1",
            red_team="Gen.G"
        )
        print("Prediction successful!")
        if 'match_winner' in result:
             print(f"Winner: {result['match_winner'].get('winner')}")
        else:
             print("Key 'match_winner' missing in result.")
             
        # Check if Kai'Sa was processed correctly in features?
        # That's hard to check from outside, but if no crash, it's good sign.
        
    except Exception as e:
        print(f"!!! CRASH with Kai'Sa: {e}")
        import traceback
        traceback.print_exc()

def find_kog():
    print("\nSearching for 'Kog' in matches.csv...")
    try:
        df = pd.read_csv('matches.csv')
        champions = df['champion'].unique()
        matches = [c for c in champions if isinstance(c, str) and 'kog' in c.lower()]
        print(f"Matches for 'kog': {matches}")
        
        matches_kha = [c for c in champions if isinstance(c, str) and 'kha' in c.lower()]
        print(f"Matches for 'kha': {matches_kha}")

    except Exception as e:
        print(f"Error reading CSV: {e}")

if __name__ == "__main__":
    check_backend()
    find_kog()
