import json
import os
import itertools
from collections import defaultdict

class RoleSorter:
    def __init__(self, stats_path=None):
        if stats_path is None:
            stats_path = os.path.join(os.path.dirname(__file__), 'role_stats.json')
        
        try:
            with open(stats_path, 'r') as f:
                self.role_stats = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Role stats file not found at {stats_path}. Role sorting will be random.")
            self.role_stats = {}

        self.roles = ['top', 'jng', 'mid', 'bot', 'sup']

    def get_role_score(self, champion, role):
        """Returns the number of times a champion played a role."""
        champ_stats = self.role_stats.get(champion.strip().title(), {})
        return champ_stats.get(role, 0)

    def sort(self, champions):
        """
        Assigns a list of 5 champions to the 5 roles [top, jng, mid, bot, sup]
        to maximize the total historical play count.
        """
        if len(champions) != 5:
            # If not 5 champions, just return as is (padded or truncated)
            return champions 

        best_perm = champions
        max_score = -1

        # Brute force all 120 permutations (5!)
        for perm in itertools.permutations(champions):
            current_score = 0
            for i, role in enumerate(self.roles):
                current_score += self.get_role_score(perm[i], role)
            
            if current_score > max_score:
                max_score = current_score
                best_perm = perm
        
        return list(best_perm)

    def sort_teams(self, blue_picks, red_picks):
        """Returns sorted (blue_picks, red_picks)"""
        return self.sort(blue_picks), self.sort(red_picks)
