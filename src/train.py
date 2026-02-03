import os
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocess import run_pipeline, get_feature_lists

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'loan.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])


def build_models(preprocessor):
    return {
        'Logistic Regression': Pipeline([
            ('preprocessing', preprocessor),
            ('model', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'))
        ]),
        'Decision Tree': Pipeline([
            ('preprocessing', preprocessor),
            ('model', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
        ]),
        'Random Forest': Pipeline([
            ('preprocessing', preprocessor),
            ('model', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced'))
        ]),
    }


def tune_logistic_regression(pipeline, X_train, y_train):
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10],
        'model__solver': ['liblinear', 'lbfgs']
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}  |  CV AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train(nrows=None):
    X, y = run_pipeline(DATA_PATH, nrows=nrows)
    numeric_features, categorical_features = get_feature_lists()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor)

    print("\nTuning Logistic Regression...")
    models['Logistic Regression'] = tune_logistic_regression(
        models['Logistic Regression'], X_train, y_train
    )

    for name in ['Decision Tree', 'Random Forest']:
        print(f"Training {name}...")
        models[name].fit(X_train, y_train)

    bundle = {
        'models': models,
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train,
        'feature_names': numeric_features + categorical_features
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)

    print(f"\nDone. Models saved to {MODEL_PATH}")
    return bundle


if __name__ == '__main__':
    train(nrows=200000)
