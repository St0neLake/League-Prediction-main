# Architecture Documentation

## Overview
The **League Prediction** project is a Machine Learning-powered web application designed to predict the outcomes of League of Legends esports matches. It leverages historical match data, team performance metrics, and role-based champion compositions to forecast match winners, game duration, kills, and objectives.

## Technology Stack

### Backend & Application Logic
- **Language**: Python 3.x
- **Web Framework**: Flask (Web UI and API)
- **ML & Data Libraries**:
  - **Scikit-learn**: Pipelines and custom Transformers.
  - **LightGBM**: Gradient boosting for model training.
  - **Category Encoders**: High-cardinality encoding for Team Names.
  - **Pandas & NumPy**: Data manipulation.
  - **Joblib**: Model serialization.

### Frontend
- **HTML/Jinja2**: Server-side templates (`templates/`).
- **CSS**: Custom styling.
- **JavaScript**: Client-side logic for dynamic champion filtering and visual feedback.

## Project Structure

```
League-Prediction-main/
├── src/                    # [NEW] Core logic package
│   ├── __init__.py
│   ├── data.py             # Centralized data loading
│   ├── features.py         # Custom ML Transformers (RoleAwareMatchupTransformer, etc.)
│   ├── roles.py            # Role inference and sorting logic
│   └── role_stats.json     # Statistics for champion-to-role mapping
├── app.py                  # Flask web server
├── run_all.py              # Inference entry point (UnifiedPredictor)
├── server.bat              # [NEW] Script to launch the Web UI
├── run.bat                 # Script to retrain all models
├── matches.csv             # Historical match data
├── models/                 # Pre-trained .joblib models
├── templates/              # HTML Templates
└── training_scripts/       # (Conceptually grouped) best_*.py files
    ├── best_lightGBM_winner.py
    ├── best_kill_lightGBM.py
    ├── best_time_lightGBM.py
    └── best_turret_lightGBM.py
```

## Core Components

### 1. Source Package (`src/`)
Refactored modular logic:
-   **`features.py`**: Contains `TeamStrengthEncoder`, `ChampionHasher`, and **`RoleAwareMatchupTransformer`**.
-   **`roles.py`**: Contains `RoleSorter`. It uses historical data (`role_stats.json`) to assign the 5 chosen champions to their most likely positions (Top, Jng, Mid, Bot, Sup).
-   **`data.py`**: Unified data loading to ensure training and inference use the exact same cleaning steps.

### 2. Prediction Engine & Models
-   **Training**: Scripts like `best_lightGBM_winner.py` train LightGBM models. They use **Role-Aware** features:
    -   Instead of comparing "Team A vs Team B" as a blob, the model aligns champions by role (Top vs Top, Mid vs Mid) before calculating matchup differences.
    -   **Sample Weighting**: Recent games are weighted higher to account for meta shifts.
-   **Inference (`run_all.py`)**: `UnifiedPredictor` loads the models and orchestrates the prediction.

### 3. Web Interface (`app.py`)
-   **Inputs**: Users select roles explicitly (Top...Support).
-   **Dynamic UI**: 
    -   Fetches champion icons from Data Dragon.
    -   Prevents duplicate selections.
    -   Validates complete teams before submission.

## Data Flow

1.  **User Input**: User fills the 10 champion slots and 2 teams in the Browser.
2.  **Role Inference**: Logic in `RoleSorter` ensures inputs are correctly aliased to roles (even if input is unordered, the backend can auto-sort, though the UI now enforces order).
3.  **Feature Transformation (`src.features`)**:
    -   `RoleAwareMatchupTransformer` sorts champions and computes lane-by-lane matchup vectors.
    -   `TeamStrengthEncoder` maps team names to historical win probabilities.
4.  **Model Prediction**: LightGBM models output probabilities for Winner, Time, Kills, etc.
5.  **Result Rendering**: Predictions and confidence intervals are displayed in `results.html`.
