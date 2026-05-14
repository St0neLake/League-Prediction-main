from run_all import UnifiedPredictor
import sys
# Import features to emulate the fix in app.py
from src.features import ChampionHasher, TeamStrengthEncoder, MatchupTransformer, RoleAwareMatchupTransformer, ChampionEmbeddingTransformer

# Backward compatibility for old pickled models
TeamEncoder = TeamStrengthEncoder
TeamTargetEncoder = TeamStrengthEncoder

# Monkey-patch sklearn to support legacy models (restore _RemainderColsList)
import sklearn.compose._column_transformer
class _RemainderColsList(list):
    pass
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# Redirect stdout to a file to capture the output including colors/errors
with open('model_debug_log.txt', 'w', encoding='utf-8') as f:
    # We want to capture the print statements from UnifiedPredictor.__init__
    # Since they use print(), we can redirect sys.stdout
    original_stdout = sys.stdout
    sys.stdout = f
    
    print("Starting UnifiedPredictor debugging...")
    try:
        predictor = UnifiedPredictor()
        print("\nInitialization complete.")
        print(f"Loaded models: {list(predictor.models.keys())}")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
    finally:
        sys.stdout = original_stdout

print("Debug finished. Check model_debug_log.txt")
