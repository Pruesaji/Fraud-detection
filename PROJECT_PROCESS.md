# Fraud Detection Project Process

## Overview
This project implements a machine learning system to detect fraudulent companies in the Thai stock market (Bangkok Stock Exchange) using financial statement data. The system analyzes Balance Sheet (BS), Profit & Loss (PNL), and Cash Flow (CF) statements to identify patterns indicative of fraud.

## Project Structure

```
Fraud-detection/
├── Data Processing Scripts
│   ├── BS_df.py              # Balance Sheet data processing
│   ├── BS_ratio_df.py        # Balance Sheet ratios calculation
│   ├── PNL_df.py             # Profit & Loss data processing
│   ├── PNL_ratio_df.py       # P&L ratios calculation
│   ├── CF_df.py              # Cash Flow data processing
│   ├── CF_ratio_df.py        # Cash Flow ratios calculation
│   └── CrossStatement_df.py  # Cross-statement ratios
│
├── Data Files
│   ├── *_df.csv              # Raw financial data
│   └── *_df_cleaned.csv      # Cleaned financial data
│
├── Model Development
│   ├── SelectedFeatures.ipynb # Feature selection & model training
│   └── model/                 # Trained models directory
│       ├── xgboost.joblib     # XGBoost model (used in production)
│       ├── best_ann_model.h5  # ANN model (Keras/TensorFlow)
│       └── best_ann_model.keras
│
└── Deployment
    ├── api.py                 # FastAPI endpoint
    ├── streamlit_ui.py        # Streamlit web UI
    └── ui.py                  # Alternative UI
```

## Data Processing Pipeline

### 1. Data Loading & Preprocessing

Each financial statement type (BS, PNL, CF) follows a similar preprocessing pipeline:

#### Constants (Defined in each script)
- `RAW_PATH`: Input CSV file path
- `CLEANED_PATH`: Output CSV file path
- `RANDOM_STATE`: 42 (for reproducibility)
- `MISSING_THR`: 0.50 (drop columns with >50% missing values)
- `TARGET_FLAG`: 1 (fraud label)

#### Preprocessing Steps

**a. Data Loading & Basic Fixes**
- Load raw CSV data from financial statements
- Remove unnecessary columns (e.g., `Unnamed: 0`)
- Parse and validate `Date` column
- Drop rows with invalid dates
- Remove sparse columns (>50% missing values)

**b. Missing Value Imputation**
- Identify numeric columns
- Impute missing values using company-specific means (grouped by `Instrument`)
- Fallback to 0 for remaining NaN values
- This approach preserves company-specific financial patterns

**c. Target Label Creation**
- Define known fraudulent companies (24 companies identified):
  ```
  ACAPm.BK, AIE.BK, AJA.BK, EA.BK, GGC.BK, GJS.BK, GL.BK,
  GSTEEL.BK, IEC.BK^G19, IFEC.BK^G24, KC.BK, NATION.BK,
  PACE.BK, POLAR.BK^A25, PRO.BK^A25, RICH.BK^H20,
  STARK.BK^I24, STELLA.BK, STOWERm.BK, TRITN.BK,
  TUCC.BK^G17, WORLD.BK^I24, EARTH.BK^I19, EASTW.BK
  ```
- Create binary `Target` column (0 = legitimate, 1 = fraud)
- Save cleaned data to CSV

### 2. Train/Validation/Test Split

**Three-tier split strategy:**
- **Test Set**: 20% of total data (stratified)
- **Validation Set**: 25% of remaining 80% = 16% of total (stratified)
- **Training Set**: 75% of remaining 80% = 64% of total

**Split proportions:**
- Train: 64%
- Validation: 16%
- Test: 20%

**Key features:**
- Stratified sampling ensures balanced fraud/legitimate ratios across splits
- Reproducible with `RANDOM_STATE=42`

### 3. Feature Normalization

**Z-Score Normalization by Company:**
```python
def zscore_by_group(train_df, apply_df, group_col, num_cols):
    # Calculate mean and std per company from training data
    # Apply normalization to preserve company-specific patterns
    # Handles zero std by replacing with 1
```

**Normalization Process:**
- Calculate statistics (mean, std) per company from **training data only**
- Apply these statistics to normalize train, validation, and test sets
- Prevents data leakage
- Preserves company-specific financial behavior patterns

**Output Files for Each Statement Type:**
- `X_train_scaled_[BS|PNL|CF].csv`
- `X_val_scaled_[BS|PNL|CF].csv`
- `X_test_scaled_[BS|PNL|CF].csv`
- `y_train_[BS|PNL|CF].csv`
- `y_val_[BS|PNL|CF].csv`
- `y_test_[BS|PNL|CF].csv`

### 4. Ratio Calculations

Separate scripts calculate financial ratios:
- **BS_ratio_df.py**: Balance sheet ratios (liquidity, leverage, etc.)
- **PNL_ratio_df.py**: Profitability and efficiency ratios
- **CF_ratio_df.py**: Cash flow ratios
- **CrossStatement_df.py**: Cross-statement ratios (e.g., cash flow to income)

These ratios often reveal fraud patterns better than raw financial values.

## Model Development

### 5. Feature Selection & Model Training (SelectedFeatures.ipynb)

The Jupyter notebook contains:

**a. Feature Engineering**
- Combines data from all financial statements
- Creates additional derived features
- Selects most informative features for fraud detection

**b. Model Training**
Two models are trained:

1. **XGBoost Model** (Production Model)
   - Gradient boosting decision tree ensemble
   - Handles imbalanced data well
   - Saved as `model/xgboost.joblib`
   - Includes both model and column names

2. **Artificial Neural Network (ANN)**
   - Deep learning model
   - Keras/TensorFlow implementation
   - Saved as `model/best_ann_model.h5` and `model/best_ann_model.keras`

**c. Model Evaluation**
- Performance metrics on validation and test sets
- Feature importance analysis
- Confusion matrices and ROC curves

## Deployment

### 6. API Service (api.py)

**FastAPI REST API:**
```python
POST /predict
Request Body: {"data": [[feature1, feature2, ..., feature10]]}
Response: {"predictions": [0, 1, ...]}
```

**Features:**
- Loads XGBoost model at startup using FastAPI lifespan context
- Accepts list of feature vectors (10 features per transaction)
- Returns fraud predictions (0 or 1)
- Automatic API documentation at `/docs`

**Starting the API:**
```bash
uvicorn api:app --reload
# Server runs on http://127.0.0.1:8000
```

### 7. User Interface (streamlit_ui.py)

**Streamlit Web Application:**
- **Input Interface**: Enter 10 comma-separated feature values
- **Data Management**: Add multiple transactions, view as table
- **API Integration**: Send data to prediction endpoint
- **Results Display**: View predictions in JSON format

**Features:**
- Interactive data entry
- Real-time validation
- Visual feedback (success/error messages)
- Clear all data functionality
- JSON format preview

**Starting the UI:**
```bash
streamlit run streamlit_ui.py
# Opens browser at http://localhost:8501
```

## Complete Workflow

### Data Science Pipeline
```
1. Raw Financial Data (BS, PNL, CF CSVs)
   ↓
2. Data Preprocessing Scripts (BS_df.py, PNL_df.py, CF_df.py)
   ↓
3. Cleaned Data + Target Labels
   ↓
4. Ratio Calculation Scripts (BS_ratio_df.py, etc.)
   ↓
5. Feature Engineering + Selection (SelectedFeatures.ipynb)
   ↓
6. Model Training (XGBoost, ANN)
   ↓
7. Model Evaluation & Selection
   ↓
8. Model Export (xgboost.joblib)
```

### Production Pipeline
```
1. User Input (10 financial features)
   ↓
2. Streamlit UI (streamlit_ui.py)
   ↓
3. HTTP POST Request
   ↓
4. FastAPI Endpoint (api.py)
   ↓
5. XGBoost Model Prediction
   ↓
6. Return Fraud Prediction (0 or 1)
   ↓
7. Display Results to User
```

## Key Technical Decisions

### 1. Company-Specific Normalization
- Financial patterns vary significantly between companies
- Normalizing within company groups preserves meaningful variations
- Prevents large companies from dominating the model

### 2. Stratified Splitting
- Fraud cases are rare (imbalanced dataset)
- Stratification ensures proportional representation in all splits
- Critical for reliable model evaluation

### 3. No Data Leakage
- Normalization statistics calculated only from training data
- Same statistics applied to validation and test sets
- Ensures realistic performance estimates

### 4. XGBoost for Production
- Better performance on tabular financial data
- Faster inference than neural networks
- Easier to deploy and maintain
- Built-in feature importance

### 5. Multiple Financial Statements
- Fraud often shows patterns across multiple statements
- Cross-statement ratios are powerful fraud indicators
- Comprehensive view of company financial health

## Running the Complete System

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Data Processing (if needed)
```bash
python BS_df.py
python PNL_df.py
python CF_df.py
python BS_ratio_df.py
python PNL_ratio_df.py
python CF_ratio_df.py
python CrossStatement_df.py
```

### Step 3: Model Training (if needed)
```bash
jupyter notebook SelectedFeatures.ipynb
# Run all cells to train models
```

### Step 4: Start API Server
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Step 5: Start UI (in separate terminal)
```bash
streamlit run streamlit_ui.py
```

### Step 6: Make Predictions
1. Open browser to Streamlit UI (usually http://localhost:8501)
2. Enter 10 comma-separated feature values
3. Click "Add Row" to add transaction
4. Click "Send to API" to get predictions
5. View fraud detection results

## Model Input Format

The model expects 10 numerical features per transaction. These are selected financial metrics and ratios that are most predictive of fraud. The exact features are defined in the trained model's column list.

## Performance Considerations

- **Imbalanced Dataset**: Fraud cases are rare, requiring special handling
- **Feature Selection**: Critical for model performance and interpretability
- **Validation Strategy**: Stratified K-fold or temporal validation recommended
- **Threshold Tuning**: May need to adjust classification threshold based on business needs

## Future Improvements

1. Add temporal features (time-series patterns)
2. Implement ensemble methods combining multiple models
3. Add explainability features (SHAP values)
4. Real-time data pipeline integration
5. Automated retraining pipeline
6. Enhanced UI with visualizations
7. Model monitoring and drift detection

## Technologies Used

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, XGBoost, TensorFlow/Keras
- **API**: FastAPI, uvicorn
- **UI**: Streamlit
- **Model Persistence**: joblib

## Notes

- All Thai comments in code (e.g., "ถึงบรรทัดสุดท้ายแล้วจ้า" = "Reached the last line")
- Financial data from Bangkok Stock Exchange (.BK suffix)
- 24 known fraudulent companies used for supervised learning
- Model requires exactly 10 features for prediction
- Supports batch predictions (multiple transactions at once)
