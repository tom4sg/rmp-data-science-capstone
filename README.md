# RMP Data Science Capstone (Professor Evaluations)

---

## Overview

This project analyzes a dataset of professor evaluations (rmpCapstone) to answer ten research questions and an extra-credit exploration. It demonstrates data cleaning, statistical testing, regression modeling, classification, and visualization techniques using Python. Results are compiled into a PDF report and accompanying Python script.

## Repository Structure

```bash
├── README.md              # Project overview and instructions
├── rmpCapstoneNum.csv     # Numeric dataset (89,893 records)
├── rmpCapstoneQual.csv    # Qualitative dataset (89,893 records)
├── analysis.py            # Python script performing analysis and plotting
└── project_report.pdf     # Final PDF report (4–6 pages with figures & results)
```

## Prerequisites

- Python 3.8+
- pip (or conda)

### Python Libraries

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

## Data Description

- **rmpCapstoneNum.csv** (no header):
  1. `avg_rating`            – Mean quality rating
  2. `avg_difficulty`        – Mean difficulty rating
  3. `num_ratings`           – Count of ratings
  4. `received_pepper`       – Boolean: judged “hot”
  5. `would_take_again_pct`  – % students who’d retake
  6. `num_online_ratings`    – Count of online-course ratings
  7. `is_male`               – Boolean: professor male
  8. `is_female`             – Boolean: professor female

- **rmpCapstoneQual.csv** (no header):
  1. `major`                 – Field of study
  2. `university`            – Institution name
  3. `state`                 – U.S. state abbreviation

## Analysis Script (`analysis.py`)

1. **Data Preprocessing**  
   - Read both CSVs, assign column names.  
   - Merge numeric & qualitative tables.  
   - Drop rows missing critical fields.  
   - Set random seed for reproducibility.

2. **Statistical Tests & Visualizations**  
   - **Q1**: Gender bias (Welch’s t-test, Mann–Whitney U, Cohen’s d)  
   - **Q2**: Experience effect via correlation, OLS, log-transform, Kruskal–Wallis  
   - **Q3**: Rating vs. difficulty (correlation, OLS)  
   - **Q4**: Online teaching impact (group split at median, Welch’s t-test, U-test)  
   - **Q5**: Rating vs. retake %, correlation & regression  
   - **Q6**: “Hotness” effect (Welch’s t-test, U-test)  

3. **Regression Modeling**  
   - **Q7**: Linear regression (`avg_difficulty` → `avg_rating`)  
   - **Q8**: Ridge regression (all numeric predictors, L2 regularization)  

4. **Classification Modeling**  
   - **Q9**: Logistic regression (`avg_rating` only), balanced sampling, ROC/AUC  
   - **Q10**: Logistic regression (all features), class balancing, ROC/AUC  

5. **Extra Credit**  
   - Defined STEM vs. Humanities majors, compared ratings (Welch’s t-test, U-test).

## Generating the Report

```bash
python analysis.py
# Review console outputs and generated plots
# Compile results into project_report.pdf via LaTeX, Word, or other tools.
```

---

**Author:** Tomas Gutierrez  
**Date:** April 2025
