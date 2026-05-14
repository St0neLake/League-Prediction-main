#!/usr/bin/env python
# Quick script to test if total_kills model can load
from joblib import load
import os

model_path = os.path.join('models', 'total_kills', 'total_kills_model.joblib')
print(f"Trying to load: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    try:
        model = load(model_path)
        print(f"✓ Successfully loaded total_kills model!")
        print(f"Model type: {type(model)}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✗ Model file does not exist!")
