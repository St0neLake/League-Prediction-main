import pandas as pd
import sys

def check_names():
    try:
        df = pd.read_csv('matches.csv')
    except FileNotFoundError:
        print("matches.csv not found")
        return

    champions = df['champion'].unique()
    apostrophe_champs = [c for c in champions if isinstance(c, str) and "'" in c]
    
    with open('verify_log.txt', 'w', encoding='utf-8') as f:
        f.write(f"Found {len(apostrophe_champs)} champions with apostrophes:\n")
        for c in apostrophe_champs:
            titled = c.strip().title()
            match = c == titled
            f.write(f"Original: '{c}' | Titled: '{titled}' | Match: {match}\n")
            if not match:
                f.write(f"!!! MISMATCH DETECTED for {c}\n")

        # Also check specific problematic ones if not found
        known_issues = ["Kai'Sa", "Cho'Gath", "Vel'Koz", "Kog'Maw", "Rek'Sai", "Bel'Veth", "Kha'Zix"]
        f.write("\nChecking known problematic champions present in CSV:\n")
        for k in known_issues:
            if k in champions:
                titled = k.strip().title()
                f.write(f"'{k}' is in CSV. Titled: '{titled}'. Match: {k == titled}\n")
            else:
                f.write(f"'{k}' is NOT in CSV (might be spelled differently)\n")

if __name__ == "__main__":
    check_names()
