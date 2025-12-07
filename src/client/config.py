from pathlib import Path

# Base directory
BASE_DIR = Path("./")

# Data paths
DATA = BASE_DIR / "data"
TEST_FILE = DATA / "Paitients_Files_Test.csv"

# History paths
HISTORY = DATA / "history"
HISTORY_FILE = HISTORY / "history.csv"

# Model path
MODELS = BASE_DIR / "models"
BEST_MODEL = MODELS / "best_mlp.keras"
SCALER = MODELS / "scaler.joblib"



# ENV when using standalone streamlit server
ENV_PATH = Path('../../env/online.env')

# Feature names
FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points',
    'worst symmetry', 'worst fractal dimension'
]

# Feature ranges and descriptions
FEATURE_CONFIG = {
    'mean radius': {'default': 14.12, 'min': 6.9, 'max': 28.1, 'description': 'Average distance from center to perimeter'},
    'mean texture': {'default': 19.29, 'min': 9.7, 'max': 40.0, 'description': 'Standard deviation of grayscale values'},
    'mean perimeter': {'default': 91.97, 'min': 43.8, 'max': 188.6, 'description': 'Mean size of the core tumor'},
    'mean area': {'default': 654.8, 'min': 180.0, 'max': 2550.0, 'description': 'Mean area of the core tumor'},
    'mean smoothness': {'default': 0.096, 'min': 0.05, 'max': 0.16, 'description': 'Local variation in radius lengths'},
    'mean compactness': {'default': 0.104, 'min': 0.02, 'max': 0.35, 'description': 'Perimeter^2 / area - 1.0'},
    'mean concavity': {'default': 0.088, 'min': 0.0, 'max': 0.45, 'description': 'Severity of concave portions of the contour'},
    'mean concave points': {'default': 0.048, 'min': 0.0, 'max': 0.22, 'description': 'Number of concave portions of the contour'},
    'mean symmetry': {'default': 0.181, 'min': 0.1, 'max': 0.35, 'description': 'Symmetry of the cell nucleus'},
    'mean fractal dimension': {'default': 0.062, 'min': 0.04, 'max': 0.1, 'description': 'Fractal dimension approximation'},
    'radius error': {'default': 0.405, 'min': 0.1, 'max': 2.9, 'description': 'Standard error for radius'},
    'texture error': {'default': 1.216, 'min': 0.3, 'max': 4.9, 'description': 'Standard error for texture'},
    'perimeter error': {'default': 2.866, 'min': 0.7, 'max': 22.0, 'description': 'Standard error for perimeter'},
    'area error': {'default': 40.34, 'min': 6.8, 'max': 545.0, 'description': 'Standard error for area'},
    'smoothness error': {'default': 0.007, 'min': 0.001, 'max': 0.016, 'description': 'Standard error for smoothness'},
    'compactness error': {'default': 0.025, 'min': 0.002, 'max': 0.14, 'description': 'Standard error for compactness'},
    'concavity error': {'default': 0.031, 'min': 0.0, 'max': 0.4, 'description': 'Standard error for concavity'},
    'concave points error': {'default': 0.011, 'min': 0.0, 'max': 0.05, 'description': 'Standard error for concave points'},
    'symmetry error': {'default': 0.020, 'min': 0.007, 'max': 0.08, 'description': 'Standard error for symmetry'},
    'fractal dimension error': {'default': 0.003, 'min': 0.0008, 'max': 0.03, 'description': 'Standard error for fractal dimension'},
    'worst radius': {'default': 16.27, 'min': 7.9, 'max': 36.0, 'description': 'Worst (largest) radius value'},
    'worst texture': {'default': 25.68, 'min': 12.0, 'max': 60.0, 'description': 'Worst texture value'},
    'worst perimeter': {'default': 107.26, 'min': 50.0, 'max': 251.0, 'description': 'Worst perimeter value'},
    'worst area': {'default': 880.5, 'min': 185.0, 'max': 4250.0, 'description': 'Worst area value'},
    'worst smoothness': {'default': 0.132, 'min': 0.07, 'max': 0.23, 'description': 'Worst smoothness value'},
    'worst compactness': {'default': 0.254, 'min': 0.05, 'max': 1.1, 'description': 'Worst compactness value'},
    'worst concavity': {'default': 0.272, 'min': 0.0, 'max': 1.3, 'description': 'Worst concavity value'},
    'worst concave points': {'default': 0.114, 'min': 0.0, 'max': 0.3, 'description': 'Worst concave points value'},
    'worst symmetry': {'default': 0.290, 'min': 0.15, 'max': 0.66, 'description': 'Worst symmetry value'},
    'worst fractal dimension': {'default': 0.083, 'min': 0.05, 'max': 0.2, 'description': 'Worst fractal dimension value'}
}

# Custom CSS
CUSTOM_CSS = """
    <style>
    .main {
        padding-top: 0rem;
    }
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .result-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 20px;
        border-radius: 5px;
        color: #155724;
    }
    .result-danger {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 20px;
        border-radius: 5px;
        color: #721c24;
    }
    </style>
    """