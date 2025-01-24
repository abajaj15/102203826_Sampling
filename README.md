# README: 

## Project Description

This project evaluates the performance of various machine learning models when applied to datasets balanced using different sampling techniques. The goal is to identify the best combination of sampling method and machine learning model for maximum accuracy.

## Dataset

- **Input File**: `Creditcard_data.csv`
- **Target Column**: `Class`
- **Features**: All other columns

## Sampling Techniques Used

| Sampling Technique  |
| ------------------- |
| Original            |
| Random OverSampler  |
| SMOTE               |
| Random UnderSampler |
| SMOTEENN            |

## Models Used

| Model                  |
| ---------------------- |
| Random Forest          |
| Logistic Regression    |
| Support Vector Machine |
| Naive Bayes            |
| Decision Tree          |

## Output

The results of the sampling techniques and model evaluations will be presented in both tabular and matrix formats for clarity.
![image](https://github.com/user-attachments/assets/5d66d29a-9fda-43d2-9071-b9f3eec60faa)



###

## Instructions for Reproduction

1. Install dependencies:
   ```bash
   pip install pandas scikit-learn imbalanced-learn
   ```
2. Place `Creditcard_data.csv` in the working directory.
3. Run the provided Python script.
4. View the outputs in the console and save them for reference.

## Notes

- Ensure the dataset is properly formatted and contains no missing values.
- Adjust random states for reproducibility if necessary.
- Experiment with additional models or metrics as needed.

---

