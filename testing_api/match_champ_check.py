import requests
import time
from datetime import datetime, timezone

# --- API Details ---
API_URL = "https://esports-api.lolesports.com/persisted/gw"
FEED_URL = "https://feed.lolesports.com/livestats/v1"
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
HEADERS = {"x-api-key": API_KEY}
LEAGUE_SLUGS = ["lec", "lfl"]  # Will check for leagues in this order

def get_league_ids():
    """Finds the unique IDs for the specified leagues."""
    ids = []
    print(f"Searching for league IDs: {', '.join(s.upper() for s in LEAGUE_SLUGS)}")
    try:
        response = requests.get(f"{API_URL}/getLeagues", headers=HEADERS, params={"hl": "en-US"})
        response.raise_for_status()
        leagues = response.json().get("data", {}).get("leagues", [])

        for slug in LEAGUE_SLUGS:
            for league in leagues:
                if league.get("slug") == slug:
                    print(f"✅ Found {slug.upper()} league ID: {league.get('id')}")
                    ids.append(league.get('id'))
        return ids
    except requests.exceptions.RequestException as e:
        print(f"Error fetching league data: {e}")
        return []

def find_active_or_next_match(league_ids):
    """
    Finds a currently live match first. If none, finds the next upcoming match.
    """
    print("Checking for live matches...")
    try:
        # 1. Check for live games first
        live_params = {"hl": "en-US"}
        response = requests.get(f"{API_URL}/getLive", headers=HEADERS, params=live_params)
        response.raise_for_status()
        live_data = response.json()
        live_events = live_data.get("data", {}).get("schedule", {}).get("events", [])

        if live_events:
            for league_id in league_ids: # Check in priority order
                for event in live_events:
                    if (event.get("league", {}).get("id") == league_id and
                        event.get("state") == "inProgress"):
                        print("✅ Found a live match already in progress!")
                        return event

        # 2. If no live games, check the schedule for the next upcoming one
        print("No live matches found. Checking for the next upcoming match...")
        schedule_params = {"hl": "en-US", "leagueId": ",".join(league_ids)}
        response = requests.get(f"{API_URL}/getSchedule", headers=HEADERS, params=schedule_params)
        response.raise_for_status()
        schedule_data = response.json()

        events = schedule_data.get("data", {}).get("schedule", {}).get("events", [])
        for event in events:
            if event.get("state") == "unstarted":
                return event

        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching event data: {e}")
        return None

def find_new_live_game(match_id, processed_game_ids):
    """Polls for a new 'inProgress' game within a match that hasn't been processed yet."""
    try:
        params = {"hl": "en-US", "id": match_id}
        response = requests.get(f"{API_URL}/getEventDetails", headers=HEADERS, params=params)
        response.raise_for_status()
        event_details = response.json()

        match_state = event_details.get("data", {}).get("event", {}).get("state")
        games = event_details.get("data", {}).get("event", {}).get("match", {}).get("games", [])

        for game in games:
            game_id = game.get("id")
            if game.get("state") == "inProgress" and game_id not in processed_game_ids:
                return game, match_state

        return None, match_state
    except requests.exceptions.RequestException:
        return None, "inProgress"

def wait_for_game_to_finish(game_id):
    """Polls the window endpoint until the game state is 'finished'."""
    print(f"---> Waiting for Game (ID: {game_id}) to finish...")
    while True:
        try:
            window_url = f"{FEED_URL}/window/{game_id}"
            response = requests.get(window_url)
            if response.status_code == 200:
                data = response.json()
                if data.get("frames") and data["frames"][-1].get("gameState") == "finished":
                    print(f"---> Game {game_id} has finished.")
                    return
        except requests.exceptions.RequestException as e:
            print(f"Minor error while checking game finish status: {e}")

        time.sleep(60) # Check every minute

def print_champion_select(game_id, processed_game_count):
    """Fetches and prints the champion select for a given game ID."""
    try:
        window_url = f"{FEED_URL}/window/{game_id}"
        response = requests.get(window_url)
        response.raise_for_status()
        data = response.json()

        blue_team_meta = data.get("gameMetadata", {}).get("blueTeamMetadata", {})
        red_team_meta = data.get("gameMetadata", {}).get("redTeamMetadata", {})

        blue_team_name = blue_team_meta.get("esportsTeamId")
        red_team_name = red_team_meta.get("esportsTeamId")

        print("\n" + "="*45)
        print(f"🔥 CHAMPIONS LOCKED IN FOR GAME {processed_game_count + 1} 🔥")
        print("="*45)

        print(f"\n🔷 BLUE TEAM ({blue_team_name}):")
        for p in blue_team_meta.get("participantMetadata", []):
            print(f"  - {p.get('summonerName'):<16} ({p.get('championId')})")

        print(f"\n🔶 RED TEAM ({red_team_name}):")
        for p in red_team_meta.get("participantMetadata", []):
            print(f"  - {p.get('summonerName'):<16} ({p.get('championId')})")

        print("\n" + "="*45)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching live game window details: {e}")

if __name__ == "__main__":
    league_ids = get_league_ids()
    if not league_ids:
        print("Could not find any of the specified leagues. Aborting.")
    else:
        event = find_active_or_next_match(league_ids)

        if not event:
            print("No live or upcoming matches found in the schedule.")
        else:
            match = event.get('match', {})
            teams = match.get('teams', [])
            if len(teams) >= 2:
                print(f"Tracking Match: {teams[0].get('name', 'TBD')} vs {teams[1].get('name', 'TBD')}")

            if event.get("state") == "unstarted":
                start_time_str = event.get("startTime")
                start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                wait_seconds = (start_time - datetime.now(timezone.utc)).total_seconds()
                if wait_seconds > 0:
                    print(f"Match starts at {start_time.strftime('%Y-%m-%d %I:%M %p %Z')}. Waiting...")
                    time.sleep(wait_seconds)

            processed_game_ids = set()
            match_id = match.get("id")

            while True:
                live_game, match_state = find_new_live_game(match_id, processed_game_ids)

                if live_game:
                    game_id = live_game.get("id")
                    print_champion_select(game_id, len(processed_game_ids))
                    processed_game_ids.add(game_id)
                    wait_for_game_to_finish(game_id)

                elif match_state == 'completed':
                    print("✅ Match series has concluded.")
                    break
                else:
                    print("Waiting for the next game to start...")
                    time.sleep(30)