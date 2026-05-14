import requests
import time
from datetime import datetime

# --- API Details ---
API_URL = "https://esports-api.lolesports.com/persisted/gw"
FEED_URL = "https://feed.lolesports.com/livestats/v1"
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
HEADERS = {"x-api-key": API_KEY}

def get_live_lec_match_id():
    """Finds the matchId of the first currently live LEC match."""
    print("Searching for live LEC matches...")
    try:
        params = {"hl": "en-US"}
        response = requests.get(f"{API_URL}/getLive", headers=HEADERS, params=params)
        response.raise_for_status()
        live_data = response.json()

        events = live_data.get("data", {}).get("schedule", {}).get("events", [])
        if not events:
            return None

        for event in events:
            if event.get("league", {}).get("slug") == "lec" and event.get("state") == "inProgress":
                match_id = event.get("match", {}).get("id")
                print(f"✅ Found live LEC match. Match ID: {match_id}")
                return match_id
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error checking for live matches: {e}")
        return None

def get_in_progress_game_id(match_id):
    """Gets the specific gameId for a game that is in progress from a matchId."""
    print(f"Fetching event details to find the active gameId for match {match_id}...")
    try:
        params = {"hl": "en-US", "id": match_id}
        response = requests.get(f"{API_URL}/getEventDetails", headers=HEADERS, params=params)
        response.raise_for_status()
        event_details = response.json()

        games = event_details.get("data", {}).get("event", {}).get("match", {}).get("games", [])
        for game in games:
            if game.get("state") == "inProgress":
                game_id = game.get("id")
                print(f"✅ Found in-progress game. Game ID: {game_id}")
                return game_id
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching event details: {e}")
        return None

def poll_game_details(game_id):
    """Polls the /details endpoint every 15 seconds for live game updates."""
    print("\n--- Starting Live Game Update Feed (polling every 15s) ---")

    # Get initial metadata to map participantId to summonerName
    window_url = f"{FEED_URL}/window/{game_id}"
    initial_response = requests.get(window_url)
    initial_data = initial_response.json()

    blue_team = initial_data.get("gameMetadata", {}).get("blueTeamMetadata", {})
    red_team = initial_data.get("gameMetadata", {}).get("redTeamMetadata", {})

    player_map = {}
    for participant in blue_team.get("participantMetadata", []):
        player_map[participant['participantId']] = participant['summonerName']
    for participant in red_team.get("participantMetadata", []):
        player_map[participant['participantId']] = participant['summonerName']

    while True:
        try:
            response = requests.get(f"{FEED_URL}/details/{game_id}")
            response.raise_for_status()
            data = response.json()

            if not data.get("frames"):
                print("Waiting for game data to become available...")
                time.sleep(15)
                continue

            latest_frame = data["frames"][-1]
            rfc_time = latest_frame.get("rfc460Timestamp")
            game_time = datetime.fromisoformat(rfc_time.replace("Z", "+00:00")).strftime('%H:%M:%S')

            print(f"\n--- Update at {game_time} UTC ---")

            for participant in latest_frame.get("participants", []):
                participant_id = participant.get("participantId")
                summoner_name = player_map.get(participant_id, f"Participant {participant_id}")

                print(
                    f"  - {summoner_name:<16} | "
                    f"Lvl: {participant.get('level', 0):<2} | "
                    f"KDA: {participant.get('kills', 0)}/{participant.get('deaths', 0)}/{participant.get('assists', 0)} | "
                    f"CS: {participant.get('creepScore', 0):<3} | "
                    f"Gold: {participant.get('totalGold', 0)}"
                )

            # Check if game is finished
            # Note: The 'details' endpoint doesn't have 'gameState', so we check the 'window' endpoint.
            window_response = requests.get(window_url)
            window_data = window_response.json()
            if window_data["frames"][-1].get("gameState") == "finished":
                print("\n--- Game has ended ---")
                break

            # Wait for 15 seconds before the next poll. You can adjust this value.
            time.sleep(15)

        except requests.exceptions.RequestException as e:
            print(f"Error polling for details: {e}")
            break
        except KeyboardInterrupt:
            print("\nStopping polling.")
            break

if __name__ == "__main__":
    live_match_id = get_live_lec_match_id()
    if live_match_id:
        game_id = get_in_progress_game_id(live_match_id)
        if game_id:
            poll_game_details(game_id)
        else:
            print("Could not find a specific game in progress for this match.")
    else:
        print("No live LEC match found at the moment.")