#!/usr/bin/env python
"""Test script to verify UnifiedPredictor loads total_kills model correctly"""

import sys
sys.path.insert(0, '.')

from run_all import UnifiedPredictor

print("=" * 60)
print("TESTING UNIFIED PREDICTOR MODEL LOADING")
print("=" * 60)

predictor = UnifiedPredictor()

print("\nChecking which models are loaded:")
print(f"  match_winner: {'✓' if 'match_winner' in predictor.models else '✗'}")
print(f"  game_length: {'✓' if 'game_length' in predictor.models else '✗'}")
print(f"  total_towers: {'✓' if 'total_towers' in predictor.models else '✗'}")
print(f"  total_kills: {'✓' if 'total_kills' in predictor.models else '✗ MISSING!'}")
print(f"  kill_thresholds: {'✓' if 'kill_thresholds' in predictor.models else '✗'}")
print(f"  tower_thresholds: {'✓' if 'tower_thresholds' in predictor.models else '✗'}")
print(f"  time_thresholds: {'✓' if 'time_thresholds' in predictor.models else '✗'}")

print(f"\nAll models keys: {list(predictor.models.keys())}")

if 'total_kills' not in predictor.models:
    print("\n" + "!" * 60)
    print("ERROR: total_kills model is NOT loaded!")
    print("!" * 60)
    sys.exit(1)
else:
    print("\n✓ total_kills model is loaded successfully!")
    print(f"  Model type: {type(predictor.models['total_kills'])}")
    
    # Test prediction
    print("\nTesting prediction...")
    result = predictor.predict_all(
        blue_picks=["Aatrox", "Ahri", "Akali", "Alistar", "Ambessa"],
        red_picks=["Amumu", "Anivia", "Aphelios", "Ashe", "AurelionSol"],
        blue_team="Unknown",
        red_team="Unknown"
    )
    
    print(f"\nResult keys: {list(result.keys())}")
    if 'total_kills' in result:
        print(f"✓ total_kills in result: {result['total_kills']}")
    else:
        print("✗ total_kills NOT in result!")
        
print("=" * 60)
