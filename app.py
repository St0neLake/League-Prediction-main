from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from datetime import datetime
from run_all import UnifiedPredictor
# Import features to ensure they are available in __main__ namespace for pickle loading
from src.features import ChampionHasher, TeamStrengthEncoder, MatchupTransformer, RoleAwareMatchupTransformer, ChampionEmbeddingTransformer
from src.performance_analysis import ModelEvaluator
from src.data import load_data

# Monkey-patch sklearn to support legacy models (restore _RemainderColsList)
import sklearn.compose._column_transformer
class _RemainderColsList(list):
    pass
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# Backward compatibility for old pickled models
TeamEncoder = TeamStrengthEncoder
TeamTargetEncoder = TeamStrengthEncoder

app = Flask(__name__)
predictor = UnifiedPredictor()

# Load data from matches.csv
try:
    matches_df = pd.read_csv('matches.csv')
except FileNotFoundError:
    print("Error: matches.csv not found. Make sure it's in the same directory.")
    exit()

# Extract unique leagues
leagues = sorted(matches_df['league'].unique().tolist())
# Extract unique champions, filtering out non-string values
champions = sorted([champ for champ in matches_df['champion'].unique() if isinstance(champ, str)])

@app.route('/get_teams/<league>')
def get_teams(league):
    teams_in_league = sorted(matches_df[matches_df['league'] == league]['teamname'].unique().tolist())
    return jsonify(teams=teams_in_league)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        blue_team = request.form.get('blue_team', '').strip()
        red_team = request.form.get('red_team', '').strip()
        blue_picks = request.form.getlist('blue_picks')
        red_picks = request.form.getlist('red_picks')

        # Convert empty team names to "Unknown" to enable team-less predictions
        if not blue_team:
            blue_team = "Unknown"
        if not red_team:
            red_team = "Unknown"

        if len(blue_picks) != 5 or len(red_picks) != 5:
            error = "Please select 5 champions for each team."
            return render_template('index.html', leagues=leagues, champions=champions, error=error)

        result = predictor.predict_all(
            blue_picks=blue_picks,
            red_picks=red_picks,
            blue_team=blue_team,
            red_team=red_team
        )
        
        # Debug print to check what's in the result
        print("\n=== DEBUG: Prediction Result Keys ===")
        print("Keys in result:", list(result.keys()))
        if 'total_kills' in result:
            print("total_kills data:", result['total_kills'])
        else:
            print("WARNING: total_kills key NOT found in result!")
        print("=== END DEBUG ===\n")
        
        return render_template('results.html', result=result)
    else:
        return render_template('index.html', leagues=leagues, champions=champions, **get_page_data())

@app.route('/performance')
def performance():
    try:
        evaluator = ModelEvaluator()
        # Load processed data that has 'blue_picks', 'red_picks', etc.
        # matches_df (global) is raw data, so we need to use load_data processing
        processed_df = load_data('matches.csv')
        stats = evaluator.evaluate_on_data(processed_df)
        
        if stats:
            return render_template('performance.html', **stats)
        else:
            return render_template('performance.html', error="Could not calculate performance statistics.")
    except Exception as e:
        return render_template('performance.html', error=f"An error occurred: {str(e)}")

def get_page_data():
    """Helper function to get shared data for the index page."""
    data_file_path = 'matches.csv'
    formatted_time = "Data file not found"

    try:
        mtime = os.path.getmtime(data_file_path)
        dt_object = datetime.fromtimestamp(mtime)
        formatted_time = dt_object.strftime('%B %d, %Y at %I:%M %p')
    except FileNotFoundError:
        print(f"Warning: '{data_file_path}' not found for timestamp.")

    return {
        'data_last_updated': formatted_time
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=False)
