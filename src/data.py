import pandas as pd

def load_data(path):
    dtype = {
        'gameid': 'string',
        'side': 'string',
        'champion': 'string',
        'teamname': 'string',
        'result': 'int8',
        'date': 'string',
        'participantid': 'string',
        'kills': 'float32',
        'gamelength': 'float32'
    }
    df = pd.read_csv(
        path,
        usecols=['gameid', 'side', 'champion', 'teamname', 'result', 'date', 'participantid', 'kills', 'gamelength'],
        dtype=dtype,
        parse_dates=['date'],
        na_filter=False
    )
    matches = []
    # Optimization: Filter participants first to reduce iteration size
    df = df[~df['participantid'].isin(['100', '200'])]
    
    # Group by gameid to reconstruct matches
    for game_id, game_data in df.groupby('gameid'):
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
                'result': blue_team['result'].iloc[0],
                'date': blue_team['date'].iloc[0],
                'blue_side': 1,  # Blue side indicator (always 1 for blue)
                'total_kills': game_data['kills'].sum(),  # Sum of all player kills in the match
                'gamelength': blue_team['gamelength'].iloc[0]  # Game length in seconds
            })
        except Exception:
            continue
            
    return pd.DataFrame(matches).sort_values('date')
