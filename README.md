# League Prediction

A Machine Learning powered application to predict the outcome of League of Legends esports matches. This tool predicts the match winner, game duration, total kills, and total towers destroyed using a suite of trained LightGBM models.

## 📋 Prerequisites

- **Python 3.8+** must be installed on your system.
- **Git** (optional, for cloning).

## 🚀 Installation & Setup

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd League-Prediction-main
    ```

2.  **Create a Virtual Environment**:
    It is recommended to use a virtual environment to manage dependencies.
    ```bash
    # Windows
    python -m venv venv

    # macOS/Linux
    python3 -m venv venv
    ```

3.  **Activate the Environment**:
    ```bash
    # Windows
    .\venv\Scripts\activate

    # macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### 1. Data Setup & Training (`run.bat`)
The `run.bat` script is designed to:
1.  Download the latest `matches.csv` dataset.
2.  Run the training scripts to generating/update the models in the `models/` directory.

> **Note:** The script expects the virtual environment to be named `.venv`. If you named it `venv`, you may need to edit line 5 of `run.bat`.

Double-click `run.bat` or run it from the terminal:
```bash
.\run.bat
```

### 2. Web Application
To start the user interface where you can interactively select teams and champions:

```bash
python app.py
```
*   Open your browser and navigate to `http://localhost:9000` (or the port displayed in the terminal).
*   Select the **League**, **Blue Team**, **Red Team**, and **5 Champions** for each side.
*   Click **Predict** to see the results.

### 3. Command Line Interface (CLI)
To run a test prediction directly in your terminal without the web server:

```bash
python run_all.py
```
This will execute the `UnifiedPredictor` with a sample matchup defined in the `if __name__ == "__main__":` block at the bottom of the file.

## 📂 Project Structure

- **`app.py`**: The Flask web application entry point.
- **`run_all.py`**: Contains the `UnifiedPredictor` class and inference logic.
- **`run.bat`**: Windows automation script for data downloading and model training.
- **`models/`**: Stores the pre-trained `.joblib` models.
- **`templates/`**: HTML/Jinja2 templates for the web interface.
- **`matches.csv`**: The historical dataset used for training and getting team/champion lists.

## 🛠️ Technology Stack
- **Languages**: Python, HTML, CSS
- **ML Frameworks**: LightGBM, Scikit-Learn, Pandas
- **Web Framework**: Flask
