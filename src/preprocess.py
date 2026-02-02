import pandas as pd
import numpy as np

SELECTED_COLS = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade',
    'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
    'purpose', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'loan_status'
]

GOOD_STATUS = {'Fully Paid'}
BAD_STATUS = {
    'Charged Off', 'Default', 'Late (31-120 days)',
    'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off'
}

NUMERIC_FEATURES = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
    'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc',
    'TotalDebt', 'MonthlyIncome', 'Debt_to_Income', 'LoanToIncome'
]

CATEGORICAL_FEATURES = [
    'term', 'grade', 'emp_length', 'home_ownership',
    'verification_status', 'purpose'
]

EMP_LENGTH_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
    '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
    '8 years': 8, '9 years': 9, '10+ years': 10
}


def load_raw(path, nrows=None):
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    return df[[c for c in SELECTED_COLS if c in df.columns]]


def build_target(df):
    df = df[df['loan_status'].isin(GOOD_STATUS | BAD_STATUS)].copy()
    df['target'] = df['loan_status'].apply(lambda s: 1 if s in GOOD_STATUS else 0)
    return df.drop(columns=['loan_status'])


def clean(df):
    for col in ['int_rate', 'revol_util']:
        if df[col].dtype == object:
            df[col] = df[col].str.replace('%', '').astype(float)

    df['term'] = df['term'].str.strip()
    df['emp_length'] = df['emp_length'].map(EMP_LENGTH_MAP)
    return df


def engineer_features(df):
    df['MonthlyIncome'] = df['annual_inc'] / 12
    df['TotalDebt'] = df['revol_bal'] + df['loan_amnt']
    df['Debt_to_Income'] = df['TotalDebt'] / (df['annual_inc'] + 1)
    df['LoanToIncome'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    return df


def get_feature_lists():
    return NUMERIC_FEATURES, CATEGORICAL_FEATURES


def run_pipeline(path, nrows=None):
    df = load_raw(path, nrows)
    df = build_target(df)
    df = clean(df)
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['target']
    print(f"Dataset ready — {X.shape[0]} rows, {X.shape[1]} features")
    print(y.value_counts().to_string())
    return X, y
