# Set C: Delivery Risk Prediction

## Project Overview

This project builds a beginner-friendly machine learning workflow to predict whether a delivery will be late. It uses a generated dataset with delivery-related features and a binary target column named `late`.

## Business Problem

Late deliveries can affect customer satisfaction and delivery operations. This project demonstrates how a machine learning workflow can help identify deliveries that may need extra attention.

---
## 🎥 Project Demonstration

A complete demonstration of this project has been recorded, including a face and screen presentation. The video explains the project workflow, dataset analysis, feature engineering, implementation of all clustering algorithms, evaluation metrics, business insights, and final conclusions.

**Video Link:**  
🔗 https://drive.google.com/file/d/1b3QIWl7TCAwzuWjWQsUz6Mssj-wNunfy/view?usp=sharing

---
## Dataset

Raw dataset path: `data/raw/set_b.csv`

| Column | Description |
|---|---|
| `record_id` | Unique record identifier; excluded from model training |
| `distance` | Delivery distance feature |
| `load` | Delivery load feature |
| `traffic` | Traffic-related feature |
| `staff` | Staffing-related feature |
| `group` | Categorical feature processed using one-hot encoding |
| `late` | Target label: `1` = late, `0` = not late |

The generator creates 300 original records, adds missing values to selected numeric columns, and appends five duplicate rows for cleaning practice.

## Project Structure

```text
project/
├── data/
│   ├── raw/
│       └── set_b.csv

├── Set_c.ipynb
├── set_c_preprocessing.joblib
└── README.md
```

## Requirements

- Python 3.10 or later recommended
- Jupyter Notebook support in VS Code
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn
- tensorflow / keras (for the ANN section)
- joblib
---

Some output files are created only after the related notebook cells run.

## Tasks

### Task 1: Dataset Generation and Loading
- Generate and save the Set C dataset.
- Load the CSV file for analysis.

### Task 2: Data Preprocessing and Feature Engineering
- Report dataset shape, exact duplicates, missing-value counts, and target counts.
- Remove exact duplicates and verify 300 unique records.
- Create fit, validation, and test partitions.
- Save partition record IDs and verify that partitions do not overlap.
- Fit median imputation on the fit partition only.
- One-hot encode `group` and handle unseen categories.
- Create the engineered feature `load / (staff + 1)`.
- Fit standard scaling on fit numeric features only.
- Keep one-hot encoded group columns unscaled.
- Check transformed shapes, feature names, and finite numeric outputs.
- Save fitted preprocessing objects.

### Task 3: Model Training and Evaluation
- Train a `DummyClassifier` baseline.
- Train a `LogisticRegression` model.
- Compare accuracy, precision, recall, and F1-score.
- Print confusion matrices.
- Save test-set predictions and late-delivery probabilities.

## Installation

Install the required packages:

```bash
pip install pandas numpy scikit-learn joblib jupyter
```

## How to Run

Run commands from your project root directory.

1. Generate the dataset:

   ```bash
   python src/generate_data.py
   ```

   If your generator is saved in a different location, run it using that path.

2. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open `set_c.ipynb`.

4. Run the notebook cells from top to bottom.

Keep the notebook and `data/` folder under the same project root so the relative paths work.

## Evaluation Metrics

- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of predicted late deliveries that are actually late.
- **Recall:** Proportion of actual late deliveries correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Summarizes correct and incorrect predictions for both classes.

Precision and recall should be considered together. The appropriate balance depends on the operational cost of missing a late delivery versus checking a delivery that would not be late.

## Leakage Prevention

- `late` is the target and must not be used as an input feature.
- `record_id` is an identifier and is excluded from model training.
- Imputation, scaling, and one-hot encoding are fitted on the fit partition only.
- Validation and test data use the fitted preprocessing objects without refitting.
- Test-set statistics must not be used to fit preprocessing or model parameters.

## Outputs

Depending on which notebook cells have been run, the project may create:

- `data/raw/set_c.csv`

- `set_c_preprocessing.joblib`


## Summary

This project demonstrates an end-to-end introductory workflow for delivery-risk prediction, including data quality checks, leakage-aware preprocessing, feature engineering, and a comparison between a baseline model and Logistic Regression. The generated dataset is intended for learning and does not establish real-world model performance.

- ---

# Author

**Name:** Santosh Verma

**Project:** Mall Shopper Profiling using Unsupervised Machine Learning

**Course:** Data Science / AI/ML
