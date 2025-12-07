# Breast Cancer Prediction AI

A polished, demo-oriented Streamlit app for exploring binary classification of breast tumor samples (Benign vs Malignant) using a TensorFlow/Keras MLP. Includes interactive single-sample input, batch CSV inference, rich Plotly visuals, and a feature guide grounded in the UCI Breast Cancer dataset.

> Important: This application is for educational and demonstration purposes only. It does not provide medical advice or diagnosis. Always consult qualified healthcare professionals.

---

## Highlights

- **Interactive UI:** Single-sample sliders organized by feature category, real-time normalization bar chart, and gauge indicator for malignancy score.
- **Batch inference:** Upload a CSV with 30 features, get classification, confidence, and downloadable results.
- **Clear feature guide:** Descriptions, ranges, and defaults for all 30 UCI breast cancer features.
- **Production-friendly patterns:** Cached model loading, clean layout, and modular sections suitable for containerization.

---

## Live Demo

Check out the deployed app here: [Breast Cancer Prediction AI](https://breast-cancer-prediction-2025.streamlit.app/)

---

## Quick start

### Local setup with uv

1. **Clone the repository**

   ```bash
   git clone https://github.com/D0nG4667/Breast-Cancer-Prediction-Data-App.git
   cd Breast-Cancer-Prediction-Data-App
   ```

2. **After cloning this repo**
  Add an env folder in the root of the project. Create an env file named `offline.env` using this sample

  ```env
  # API
  API_URL=http://api:7860/api/v1/prediction?model

  ```

3. **Install uv**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Or on Windows (PowerShell):

   ```powershell
   irm https://astral.sh/uv/install.ps1 | iex
   ```

4. **Sync dependencies**

   ```bash
   uv sync
   ```

   This will create a `.venv` and install all dependencies from `pyproject.toml` and `uv.lock`.

5. **Activate the environment**

   ```bash
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

6. **Place model artifacts**
   - **Model file:** `best_mlp.keras`
   - **Scaler file:** `scaler.joblib`

   Put both files at the project root (same directory as the Streamlit script).

7. **Run the app**

   ```bash
   uv run streamlit run app.py
   ```

   Replace `app.py` with the actual filename of your Streamlit script if different.

> Tip: GPU TensorFlow is not required for this app. CPU installations are sufficient.

---

## Docker deployment

### Build and run

1. **Dockerfile**

   ```dockerfile
    FROM python:3.12-slim AS builder

    # Install uv
    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

    WORKDIR /app

    # Copy lockfile and project metadata first
    COPY uv.lock pyproject.toml ./

    # Install dependencies only (non-editable, without project source)
    RUN --mount=type=cache,target=/root/.cache/uv \
        uv sync --locked --no-install-project --no-editable

    # Copy app source
    COPY . .

    # Sync project into environment (non-editable)
    RUN --mount=type=cache,target=/root/.cache/uv \
        uv sync --locked --no-editable

    # Final stage
    FROM python:3.12-slim

    WORKDIR /app

    # Copy virtual environment from builder
    COPY --from=builder /app/.venv /app/.venv
    COPY . .

    ENV PATH="/app/.venv/bin:$PATH"

    EXPOSE 8501

    CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

   ```

2. **Build the image**

   ```bash
   docker build -t bc-prediction-ai:latest .
   ```

3. **Run the container**

   ```bash
   docker run --rm -p 8501:8501 \
     -v $(pwd)/best_mlp.keras:/app/best_mlp.keras \
     -v $(pwd)/scaler.joblib:/app/scaler.joblib \
     bc-prediction-ai:latest
   ```

> Tip: Mount model artifacts via volumes so you can update models without rebuilding the image.

---

## App structure and navigation

### Pages

- **Single prediction:**
  - **Feature categories:** Mean, Error, Worst.
  - **Controls:** Sliders with validated ranges and defaults.
  - **Visuals:** Normalized feature bar chart and a gauge showing malignancy score.
  - **Results:** Benign/Malignant card, confidence, probability breakdown, and timestamp.

- **Batch prediction:**
  - **Upload:** CSV with exactly 30 columns in the schema below.
  - **Output:** Malignancy_Score, Classification, Confidence for each row.
  - **Actions:** Download results as CSV and view summary metrics.

- **About model:**
  - **Architecture:** TensorFlow/Keras MLP, binary output.
  - **Dataset:** UCI Breast Cancer (569 samples; 357 Benign, 212 Malignant).
  - **Performance:** Accuracy, precision, recall, F1 (demo values).
  - **Preprocessing:** StandardScaler across all 30 features, stratified split.

- **Feature guide:**
  - **Details:** Description, range, and default for each feature.
  - **Categories:** Mean, Error, Worst.

### Runtime behavior

- **Model loading:** Cached via `@st.cache_resource` to avoid reloading on UI interactions.
- **Prediction threshold:** 0.5 for binary decision.
- **Confidence:** `max(p, 1-p)` displayed as a percentage.

---

## Data schema (CSV for batch mode)

Your CSV must include the following 30 columns with numeric values. Names must match exactly:

- **Mean features:**
  - mean radius
  - mean texture
  - mean perimeter
  - mean area
  - mean smoothness
  - mean compactness
  - mean concavity
  - mean concave points
  - mean symmetry
  - mean fractal dimension

- **Error features:**
  - radius error
  - texture error
  - perimeter error
  - area error
  - smoothness error
  - compactness error
  - concavity error
  - concave points error
  - symmetry error
  - fractal dimension error

- **Worst features:**
  - worst radius
  - worst texture
  - worst perimeter
  - worst area
  - worst smoothness
  - worst compactness
  - worst concavity
  - worst concave points
  - worst symmetry
  - worst fractal dimension

### Example row

```csv
mean radius,mean texture,mean perimeter,mean area,mean smoothness,mean compactness,mean concavity,mean concave points,mean symmetry,mean fractal dimension,radius error,texture error,perimeter error,area error,smoothness error,compactness error,concavity error,concave points error,symmetry error,fractal dimension error,worst radius,worst texture,worst perimeter,worst area,worst smoothness,worst compactness,worst concavity,worst concave points,worst symmetry,worst fractal dimension
14.12,19.29,91.97,654.8,0.096,0.104,0.088,0.048,0.181,0.062,0.405,1.216,2.866,40.34,0.007,0.025,0.031,0.011,0.020,0.003,16.27,25.68,107.26,880.5,0.132,0.254,0.272,0.114,0.290,0.083
```

> Tip: Values should be within the ranges shown in the Feature Guide. Out-of-range inputs may reduce model reliability.

### Generating a sample CSV from scikit-learn

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Map scikit-learn feature names to the app’s expected names if needed
# Ensure exact matching with the 30 columns listed above
df.to_csv("samples.csv", index=False)
```

---

## Model and preprocessing

- **Architecture:** Multi-Layer Perceptron (MLP) in TensorFlow/Keras.
- **Inputs:** 30 numeric features (UCI breast cancer feature set).
- **Output:** Single sigmoid probability for malignancy.
- **Scaling:** StandardScaler applied to all features prior to inference.

> Note: `best_mlp.keras` and `scaler.joblib` are required for predictions. If you retrain, ensure the scaler and feature order are identical to the app’s schema.

---

## Development notes

- **File naming:** The Streamlit script reads `best_mlp.keras` and `scaler.joblib` from the working directory.
- **Caching:** Model/scaler are cached via `@st.cache_resource` to optimize performance.
- **Visualization:** Plotly is used for bar charts and the gauge indicator.
- **Extensibility:**
  - **Confidence calibration:** Consider Platt scaling or isotonic regression if you retrain.
  - **Model metrics:** Wire in real metrics from your training runs and add confusion matrices.
  - **Schema validation:** Add CSV header checks and dtype validation for robust batch mode.
  - **CI/CD:** Containerize with Docker and deploy via Streamlit Community Cloud or your infra.

---

## Troubleshooting

- **Model/scaler not found:**
  - **Check paths:** Ensure files are at project root or update paths in `load_models()`.
  - **Permissions:** Confirm container volume mounts and file permissions.

- **CSV errors in batch mode:**
  - **Header mismatch:** Column names must match the schema exactly.
  - **Types:** Ensure all values are numeric and properly delimited.
  - **Scaling compatibility:** Use the same scaler that was fitted during training.

- **Performance issues:**
  - **Large CSVs:** Consider chunked prediction for very large batches.
  - **Server memory:** Adjust container resources or run on a machine with more RAM.

---

## Ethics and responsible use

- **Non-diagnostic:** This demo is not a clinical decision support tool.
- **Bias and data limits:** Trained on a specific dataset; performance and generalization can vary across populations and acquisition settings.
- **Professional oversight:** Always involve qualified medical practitioners for interpretation and validation.

> If you adapt this app for research or clinical workflows, follow regulatory requirements, perform rigorous validation, and maintain transparent documentation.
