# Loan Default Risk Predictor

Predicts whether a loan will be repaid or default, using real LendingClub data and three ML models with full SHAP explainability.

---

**Data**
- LendingClub dataset 
- filtered to only clear outcomes Fully Paid vs Charged Off
- 8700 labeled samples used for training and evaluation

**Features**
- 18 raw features: loan amount, interest rate, grade, income, DTI, credit history, etc.
- 4 engineered features: TotalDebt, MonthlyIncome, Debt to Income ratio, Loan to Income ratio

**Models**
- Logistic Regression (tuned with GridSearchCV)
- Decision Tree
- Random Forest

**Evaluation**
- Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrices, ROC curves, Precision-Recall curves
- 5-fold cross-validation
- SHAP for feature-level explainability

---

## Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.61 | 0.70 | 0.69 |
| Decision Tree | 0.70 | 0.81 | 0.55 |
| Random Forest | 0.79 | 0.88 | 0.69 |

Random Forest performed best overall with 88% F1 and 99% recall.

---

## Graphs

### Confusion Matrices
![Confusion Matrices](assets/confusion_matrices.png)

### ROC Curves
![ROC Curves](assets/roc_curves.png)

### Precision-Recall
![Precision-Recall](assets/precision_recall.png)

### Feature Importance
![Feature Importance](assets/feature_importance.png)

### SHAP
![SHAP Summary](assets/shap_summary.png)
