# Student Exam Scores Prediction

A Jupyter Notebook project that implements end-to-end machine learning pipelines to predict student exam scores using both custom and scikit-learn models.

## Project Structure

```
student-scores-prediction/      # project root
├── README.md                   # this file
├── requirements.txt            # Python dependencies
└── student-scores-prediction.ipynb  # main analysis & modeling notebook
```

## Overview

This notebook walks through:

1. **Exploratory Data Analysis (EDA)**: loading the data, checking distributions, handling missing values, and capping outliers.
2. **Feature Engineering & Selection**: correlation tests for numeric features and ANOVA F-tests for categoricals, followed by one-hot encoding and scaling.
3. **Model Implementation**:
   - Custom **Linear Regression** via mini-batch gradient descent.
   - Custom **k-Nearest Neighbors** regression.
   - Comparison with scikit-learn’s `LinearRegression` and `KNeighborsRegressor`.
4. **Hyperparameter Tuning**: manual sweeps over learning rate, batch size, epochs for GD, and `k` for KNN, with log-scale plots.
5. **Evaluation**: baseline and tuned model performance using MAE and R², including k-fold cross-validation.

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/student-scores-prediction.git
   cd student-scores-prediction
   ```
2. **Create and activate a virtual environment** (optional but recommended)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .\.venv\Scripts\activate   # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook student-scores-prediction.ipynb
   ```
2. **Run all cells** in order, from data loading through final evaluation.

## Results

- **Custom Mini-Batch GD LinearRegression**: MAE ≈ 1.16, R² ≈ 0.64
- **Custom KNNRegression (k=30)**: MAE ≈ 1.34, R² ≈ 0.60
- **scikit-learn LinearRegression** matched custom GD performance, validating the implementation

---

*Created by Mykhailo Rykhalskyi*

