"""Compare scikit-learn classifiers with 5-fold stratified cross-validation on the built-in breast cancer dataset."""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODELS = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "decision_tree": DecisionTreeClassifier(random_state=0),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=0),
}


def run(seed=0):
    X, y = load_breast_cancer(return_X_y=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return {n: cross_val_score(m, X, y, cv=cv, scoring="accuracy") for n, m in MODELS.items()}


if __name__ == "__main__":
    print(f"{'model':<22}{'mean acc':>9}{'std':>8}")
    for n, s in sorted(run().items(), key=lambda kv: -kv[1].mean()):
        print(f"{n:<22}{s.mean():>9.4f}{s.std():>8.4f}")
