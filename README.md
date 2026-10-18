# health-insurance-cost-prediction
A machine learning regression project to predict health insurance charges using demographic and lifestyle features. Includes data preprocessing, exploratory data analysis, feature encoding, model training, and evaluation using R², MAE, and RMSE.

## Overview

Using a dataset of 1,338 insurance records, this project explores how factors such as age, BMI, smoking status, and region relate to medical charges, and trains a regression model to predict them.

## Dataset

The dataset (`insurance.csv`) contains 1,338 rows and 7 columns with no missing values.

| Column     | Description                                         |
|------------|-----------------------------------------------------|
| `age`      | Age of the primary beneficiary                      |
| `sex`      | Gender (female / male)                              |
| `bmi`      | Body mass index                                     |
| `children` | Number of dependents covered by insurance           |
| `smoker`   | Smoking status (yes / no)                           |
| `region`   | Residential area (northeast, northwest, southeast, southwest) |
| `charges`  | Medical costs billed by insurance (target variable) |

## Project Workflow

1. **Data loading and inspection**: shape, data types, missing values, summary statistics.
2. **Preprocessing**: binary encoding of `sex` (female = 1, male = 0) and `smoker` (yes = 1, no = 0).
3. **Exploratory data analysis**: correlation heatmap and a BMI vs. charges scatter plot split by smoking status.
4. **Feature engineering**: one-hot encoding of `region`.
5. **Train/test split**: 70% training, 30% testing (`random_state=42`).
6. **Modeling**: `LinearRegression` from scikit-learn.
7. **Evaluation**: R², MAE, and RMSE on the test set.

## Key Findings

- Smoking status has a strong positive correlation with charges (**0.79**), making it the most influential feature in the dataset.
- Smokers generally have noticeably higher medical charges than non-smokers, particularly at higher BMI values.

> Correlation does not imply causation; these results describe patterns in this dataset only.

## Results

| Metric    | Value      |
|-----------|------------|
| R² Score  | 0.77       |
| MAE       | 4,145.45   |
| RMSE      | 5,812.10   |

The model explains roughly 77% of the variance in insurance charges, with predictions off by about 4,145 on average.

## Tech Stack

- Python 3
- NumPy, pandas
- Matplotlib, seaborn
- scikit-learn
- Jupyter Notebook

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/alhinarabia/health-insurance-cost-prediction.git
   cd health-insurance-cost-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Add the dataset**: place `insurance.csv` in the project root (same folder as the notebook).

4. **Run the notebook**
   ```bash
   jupyter notebook "Insurance data Regression.ipynb"
   ```

## Project Structure

```
.
├── Insurance data Regression.ipynb   # Main analysis and model
├── insurance.csv                     # Dataset
└── README.md
```

## Possible Improvements

- Apply a log transform to `charges` to reduce skewness.
- Add interaction features (e.g., `smoker × bmi`) since the relationship differs by smoking status.
- Analyze and handle outliers.
- Compare with other models such as Ridge/Lasso, Random Forest, or Gradient Boosting.
- Use cross-validation for more robust evaluation.

## Author

Alhin Arabia: [GitHub](https://github.com/alhinarabia)
