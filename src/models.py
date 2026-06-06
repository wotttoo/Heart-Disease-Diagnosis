import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeModel:
    """Wrapper around DecisionTreeClassifier."""

    def __init__(self, random_state) -> None:
        self.model = DecisionTreeClassifier(random_state=random_state)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'DecisionTreeModel':
        self.model.fit(X_train, y_train)
        return self
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, split_name: str = 'Val') -> float:
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f'Decision Tree - {split_name} accuracy: {acc:.4f}')
        print(classification_report(y, y_pred))
        
        return acc

class KNNModel:
    """Wrapper around KNeighborsClassifier."""

    def __init__(self, n_neighbors: int = 5) -> None:
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'KNNModel':
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X: pd.DataFrame, y: pd.Series, split_name: str = 'Val') -> float:
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f'KNN - {split_name} accuracy: {acc:.4f}')
        print(classification_report(y, y_pred))

        return acc

class NaiveBayesModel:
    """Thin wrapper around GaussianNB with a consistent evaluate interface."""

    def __init__(self) -> None:
        self.model = GaussianNB()
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'NaiveBayesModel':
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X: pd.DataFrame, y: pd.Series, split_name: str = 'Val') -> float:
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f'Naive Bayes - {split_name} accuracy: {acc:.4f}')
        print(classification_report(y, y_pred))

        return acc

class EnsembleModel:
    """Stacking classifier: DT + KNN + NB base learners, KNN meta-learner.

    Parameters
    ----------
    random_state : Seed passed to DecisionTreeClassifier for reproducibility.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: StackingClassifier | None = None

    def build(self) -> 'EnsembleModel':
        self.model = StackingClassifier(
            estimators = [
                ('dt', DecisionTreeClassifier(random_state=42)),
                ('knn', KNeighborsClassifier()),
                ('nb', GaussianNB())
            ],
            final_estimator=KNeighborsClassifier(),
            stack_method='predict_proba',
            passthrough=False
        )
        return self

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'EnsembleModel':
        if self.model is None:
            self.build()
        
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X: pd.DataFrame, y: pd.Series, split_name: str = 'Val') -> float:
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f'Ensemble - {split_name} accuracy: {acc:.4f}')
        print(classification_report(y, y_pred))

        return acc