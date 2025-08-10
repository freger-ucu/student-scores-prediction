# Student Exam Scores Prediction

This project predicts student exam scores using both **custom-implemented models** and scikit-learn baselines. It includes **data preprocessing, exploratory analysis, model training, hyperparameter tuning, and evaluation**.

## Problem Statement
With the proliferation of data-driven methodologies in education, institutions are turning to predictive analytics to support student achievement and optimize resource allocation. Underperforming students face diminished academic and career prospects, and schools incur significant costs addressing remediation and dropout prevention. Early and accurate prediction of exam performance is therefore essential to enable timely, targeted interventions.

In this project, we develop a machine learning pipeline that leverages features such as hours studied, attendance rates, and parental factors to forecast student exam scores with high precision. Our objective is to provide educators with actionable insights that facilitate personalized support and improve overall educational outcomes.

#### In this project:

1. Exploratory Data Analysis will be performed to figure out features with the most correlation to student exam scores. 

2. Custom **Linear Regression** (Mini-batch Gradient Descent) and **k-Nearest Neighbors** Regression models will be implemented **from scratch** to predict student performance.

3. Models' hyperparameters will be tuned to achieve the ultimate prediction performance. 

4. Both models will be benchmarked using k-fold cross-validation via mean absolute error (MAE) and R² to determine the top performer.


## 📂 Project Structure

```
Student scores prediction/
├── data/                           # Dataset storage
│   └── students_performance.zip
├── notebooks/                      # Jupyter notebooks
│   ├── EDA_and_Preprocessing.ipynb
│   └── Tuning_and_Benchmarks.ipynb
├── src/                            # Custom Python modules
│   ├── __init__.py
│   ├── preprocess.py               # Data loading & preprocessing
│   ├── models.py                   # Custom LinearRegression & KNN
│   └── evaluation.py               # Metrics & cross-validation
├── README.md                       # Project overview and usage guide
├── requirements.txt                # Dependencies
```

## 📊 Dataset

Data: [Kaggle – Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)\
The dataset contains demographic, behavioral, and academic details of students, along with their exam scores.

### Key Features:

- **Numerical:** Attendance, Hours\_Studied
- **Categorical:** Parental\_Involvement, Access\_to\_Resources, Parental\_Education\_Level, Peer\_Influence, Learning\_Disabilities (encoded via one-hot encoding)

## 🚀 Quick Start in Google Colab

You can run the **entire project in one place** using this Colab notebook:\
[Open in Google Colab](https://colab.research.google.com/drive/14Le9Ehw26GnCMng7HzWH49bEolo2yMQu?usp=sharing)&#x20;

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/student-scores-prediction.git
cd student-scores-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 📓 How to Use

1. **EDA & Preprocessing:** `notebooks/EDA_and_Preprocessing.ipynb`
2. **Model Tuning & Benchmarks:** `notebooks/Tuning_and_Benchmarks.ipynb`
3. **Custom Python Modules:**

```python
import numpy as np
from src.preprocess import load_data, preprocess_data
from src.models import LinearRegression, KNNRegression
from src.evaluation import evaluate

X, y = preprocess_data(load_data("data/students_performance.zip"))
models = {
    "LR": LinearRegression(learning_rate=1e-3, epochs=1000, batch_size=32),
    "KNN": KNNRegression(k=30)
}
k = 5

mae_scores, r_squared = evaluate(models, k, X, y)
for name in models:
    print(f"--- {name} ---")
    print(f"Average MAE across {k} folds: {np.mean(mae_scores[name]):.4f}")
    print(f"Average R^2 across {k} folds: {np.mean(r_squared[name]):.3f}\n")
```

## 📈 Results

| Model                                 | MAE     | R²    |
|---------------------------------------| ------- | ----- |
| Custom Mini-Batch GD LinearRegression | \~1.199 | 0.636 |
| Custom K-NN Regression (k=30)         | \~1.369 | 0.596 |

**Best Model:** Linear Regression for balance of accuracy and interpretability.

## 📝 License

MIT License.

