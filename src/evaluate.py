import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
plt.style.use('seaborn-v0_8')


def load_bundle():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def save_fig(name):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    plt.savefig(os.path.join(ASSETS_DIR, name), bbox_inches='tight', dpi=150)


def metrics_table(models, X_test, y_test):
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4),
        })
    return pd.DataFrame(rows).set_index('Model')


def plot_confusion_matrices(models, X_test, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices", y=1.02)
    plt.tight_layout()
    save_fig('confusion_matrices.png')
    plt.show()


def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(7, 5))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], '--', color='grey', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    save_fig('roc_curves.png')
    plt.show()


def plot_precision_recall(models, X_test, y_test):
    plt.figure(figsize=(7, 5))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    save_fig('precision_recall.png')
    plt.show()


def plot_feature_importance(model, top_n=15):
    rf = model.named_steps['model']
    preprocessor = model.named_steps['preprocessing']

    try:
        ohe_features = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out()
        num_features = preprocessor.transformers_[0][2]
        all_features = list(num_features) + list(ohe_features)
    except Exception:
        all_features = [f"f{i}" for i in range(len(rf.feature_importances_))]

    feat_df = pd.DataFrame({'Feature': all_features, 'Importance': rf.feature_importances_})
    feat_df = feat_df.sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df)
    plt.title(f"Top {top_n} Features (Random Forest)")
    plt.tight_layout()
    save_fig('feature_importance.png')
    plt.show()


def run_shap(model, X_test):
    try:
        import shap

        rf = model.named_steps['model']
        preprocessor = model.named_steps['preprocessing']
        X_transformed = preprocessor.transform(X_test)

        try:
            ohe_features = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out()
            num_features = preprocessor.transformers_[0][2]
            feature_names = list(num_features) + list(ohe_features)
        except Exception:
            feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

        X_df = pd.DataFrame(X_transformed, columns=feature_names)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_df)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        shap.summary_plot(sv, X_df, plot_type="bar", max_display=15, show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        save_fig('shap_summary.png')
        plt.show()

        shap.summary_plot(sv, X_df, max_display=15, show=False)
        plt.title("SHAP Beeswarm")
        plt.tight_layout()
        save_fig('shap_beeswarm.png')
        plt.show()

        shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, list)
            else explainer.expected_value,
            sv[0], X_df.iloc[0], matplotlib=True, show=False
        )
        plt.title("SHAP Force Plot — Sample 0")
        plt.tight_layout()
        save_fig('shap_force.png')
        plt.show()

        print("SHAP plots saved to assets/")

    except ImportError:
        print("Run: pip install shap")


def evaluate():
    bundle = load_bundle()
    models = bundle['models']
    X_test = bundle['X_test']
    y_test = bundle['y_test']
    X_train = bundle['X_train']
    y_train = bundle['y_train']

    print("\nModel Comparison")
    print(metrics_table(models, X_test, y_test))

    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_precision_recall(models, X_test, y_test)
    plot_feature_importance(models['Random Forest'])

    print("\nCross-Validation (Random Forest, ROC-AUC)")
    cv = cross_val_score(models['Random Forest'], X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"Scores: {cv.round(4)}  |  Mean: {cv.mean():.4f}")

    print("\nRunning SHAP...")
    run_shap(models['Random Forest'], X_test.iloc[:500])


if __name__ == '__main__':
    evaluate()
